using System;
using System.Text;
using UnityEngine;
using NativeWebSocket;

public class ARIABridgeClient : MonoBehaviour
{
    [SerializeField] private string laptopIP = "192.168.1.100";
    [SerializeField] private int port = 8765;

    private WebSocket _ws;

    async void Start()
    {
        string uri = $"ws://{laptopIP}:{port}";
        Debug.Log($"[ARIA] Connecting to {uri}...");

        _ws = new WebSocket(uri);

        _ws.OnOpen += () => {
            Debug.Log("[ARIA] WebSocket OPEN");
            SendHandshake();
        };

        _ws.OnMessage += (bytes) => {
            string message = Encoding.UTF8.GetString(bytes);
            Debug.Log($"[ARIA] Echo received: {message}");
        };

        _ws.OnError += (e) => Debug.LogError($"[ARIA] Error: {e}");
        _ws.OnClose += (e) => Debug.Log($"[ARIA] Closed: {e}");

        await _ws.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (_ws != null) _ws.DispatchMessageQueue();
#endif
    }

    async void SendHandshake()
    {
        if (_ws.State != WebSocketState.Open) return;
        string msg = "{\"type\":\"handshake\",\"client\":\"aria-unity\",\"stage\":1}";
        await _ws.SendText(msg);
        Debug.Log($"[ARIA] Sent: {msg}");
    }

    async void OnApplicationQuit()
    {
        if (_ws != null) await _ws.Close();
    }
}