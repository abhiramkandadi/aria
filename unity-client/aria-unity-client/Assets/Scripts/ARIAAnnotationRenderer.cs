using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// ARIA Stage 4 — Annotation Renderer
/// Reads mask_center_2d from the bridge server response,
/// raycasts into the Quest 3 scene mesh via OVRRaycastHit,
/// and places a world-locked arrow + text label at the 3D hit point.
/// The annotation stays locked in world space as the user moves their head.
/// </summary>
public class ARIAAnnotationRenderer : MonoBehaviour
{
    [Header("Prefabs")]
    [SerializeField] private GameObject arrowPrefab;      // Arrow pointing at object
    [SerializeField] private GameObject labelPrefab;      // TextMeshPro world-space label

    [Header("Scene References")]
    [SerializeField] private Camera passthroughCamera;    // Main XR camera
    [SerializeField] private OVRSceneManager sceneManager; // For scene mesh availability

    [Header("Tuning")]
    [SerializeField] private float labelYOffset    = 0.15f;  // Metres above hit point
    [SerializeField] private float annotationScale = 0.08f;  // World-space scale of arrow
    [SerializeField] private float fadeOutSeconds  = 4.0f;   // How long annotation persists
    [SerializeField] private LayerMask raycastLayers = ~0;   // All layers by default

    // Camera resolution SAM2 uses — must match what server sends frames at
    // These match the 640x480 resize in the backend pipeline
    private const int SAM2_FRAME_W = 640;
    private const int SAM2_FRAME_H = 480;

    private GameObject _currentArrow;
    private GameObject _currentLabel;
    private Coroutine  _fadeCoroutine;

    // Last known world position — used for fallback if raycast misses
    private Vector3    _lastHitPoint = Vector3.zero;
    private bool       _hasHitPoint  = false;

    void Awake()
    {
        if (passthroughCamera == null)
            passthroughCamera = Camera.main;
    }

    /// <summary>
    /// Called by ARIABridgeClient when a new response arrives.
    /// </summary>
    public void HandleResponse(ARIAResponse response)
    {
        if (response == null) return;

        // Update label text on existing annotation regardless of raycast
        UpdateLabelText(response.text);

        float[] center2d = response.mask_center_2d;
        if (center2d == null || center2d.Length < 2)
        {
            // No mask center — use last known position or skip
            if (_hasHitPoint)
                PlaceAnnotation(_lastHitPoint, response.text, response.detection_source);
            return;
        }

        // Convert SAM2 pixel coords to viewport coords [0,1]
        float vpX = center2d[0] / SAM2_FRAME_W;
        float vpY = 1.0f - (center2d[1] / SAM2_FRAME_H); // flip Y (Unity viewport Y=0 is bottom)

        // Clamp to valid viewport range
        vpX = Mathf.Clamp01(vpX);
        vpY = Mathf.Clamp01(vpY);

        // Try OVRRaycast into scene mesh first (most accurate for MR)
        Vector3 hitPoint;
        bool hit = TryOVRRaycast(vpX, vpY, out hitPoint);

        if (!hit)
        {
            // Fallback: standard Physics raycast
            hit = TryPhysicsRaycast(vpX, vpY, out hitPoint);
        }

        if (!hit)
        {
            // Final fallback: project to a fixed depth in camera space
            hitPoint = passthroughCamera.ViewportToWorldPoint(
                new Vector3(vpX, vpY, 1.5f) // 1.5m in front of camera
            );
        }

        _lastHitPoint = hitPoint;
        _hasHitPoint  = true;

        PlaceAnnotation(hitPoint, response.text, response.detection_source);
    }

    bool TryOVRRaycast(float vpX, float vpY, out Vector3 hitPoint)
    {
        hitPoint = Vector3.zero;

        try
        {
            // Build ray from camera through viewport point
            Ray ray = passthroughCamera.ViewportPointToRay(new Vector3(vpX, vpY, 0f));

            // OVRRaycastHit queries the Quest 3 scene mesh (walls, floors, objects)
            OVRSceneRoom room = FindObjectOfType<OVRSceneRoom>();
            if (room == null) return false;

            // Use OVR's built-in scene raycast
            OVRSceneAnchor[] anchors = FindObjectsOfType<OVRSceneAnchor>();
            foreach (var anchor in anchors)
            {
                var colliders = anchor.GetComponentsInChildren<Collider>();
                foreach (var col in colliders)
                {
                    RaycastHit info;
                    if (col.Raycast(ray, out info, 10f))
                    {
                        hitPoint = info.point;
                        Debug.Log($"[ARIA Renderer] OVR scene mesh hit at {hitPoint} (anchor: {anchor.name})");
                        return true;
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[ARIA Renderer] OVRRaycast error: {e.Message}");
        }

        return false;
    }

    bool TryPhysicsRaycast(float vpX, float vpY, out Vector3 hitPoint)
    {
        hitPoint = Vector3.zero;
        Ray ray = passthroughCamera.ViewportPointToRay(new Vector3(vpX, vpY, 0f));
        RaycastHit hit;
        if (Physics.Raycast(ray, out hit, 10f, raycastLayers))
        {
            hitPoint = hit.point;
            Debug.Log($"[ARIA Renderer] Physics raycast hit at {hitPoint} ({hit.collider.name})");
            return true;
        }
        return false;
    }

    void PlaceAnnotation(Vector3 worldPos, string text, string source)
    {
        // Cancel existing fade
        if (_fadeCoroutine != null)
            StopCoroutine(_fadeCoroutine);

        // Place or move arrow
        if (_currentArrow == null && arrowPrefab != null)
            _currentArrow = Instantiate(arrowPrefab);

        if (_currentArrow != null)
        {
            _currentArrow.transform.position   = worldPos;
            _currentArrow.transform.localScale  = Vector3.one * annotationScale;
            // Arrow faces camera
            _currentArrow.transform.LookAt(passthroughCamera.transform);
            _currentArrow.transform.Rotate(0, 180f, 0);
            SetVisible(_currentArrow, true);
        }

        // Place or move label (y+offset above hit point)
        if (_currentLabel == null && labelPrefab != null)
            _currentLabel = Instantiate(labelPrefab);

        if (_currentLabel != null)
        {
            Vector3 labelPos = worldPos + Vector3.up * labelYOffset;
            _currentLabel.transform.position = labelPos;
            _currentLabel.transform.LookAt(passthroughCamera.transform);
            _currentLabel.transform.Rotate(0, 180f, 0);
            SetVisible(_currentLabel, true);
            UpdateLabelText(text);
        }

        string sourceTag = source == "sam2" ? "[SAM2]" : "[YOLO]";
        Debug.Log($"[ARIA Renderer] Annotation placed at {worldPos} {sourceTag} — '{text}'");

        // Start fade timer
        _fadeCoroutine = StartCoroutine(FadeOutAfter(fadeOutSeconds));
    }

    void UpdateLabelText(string text)
    {
        if (_currentLabel == null) return;

        // Support both TextMeshPro and legacy Text
        var tmp = _currentLabel.GetComponentInChildren<TMPro.TextMeshPro>();
        if (tmp != null) { tmp.text = text; return; }

        var legacy = _currentLabel.GetComponentInChildren<UnityEngine.UI.Text>();
        if (legacy != null) legacy.text = text;
    }

    IEnumerator FadeOutAfter(float seconds)
    {
        yield return new WaitForSeconds(seconds);
        if (_currentArrow != null) SetVisible(_currentArrow, false);
        if (_currentLabel != null) SetVisible(_currentLabel, false);
    }

    void SetVisible(GameObject go, bool visible)
    {
        if (go != null) go.SetActive(visible);
    }

    void OnDestroy()
    {
        if (_currentArrow != null) Destroy(_currentArrow);
        if (_currentLabel != null) Destroy(_currentLabel);
    }
}