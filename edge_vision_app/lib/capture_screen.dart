import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class CaptureScreen extends StatefulWidget {
  final String serverIp;

  const CaptureScreen({super.key, required this.serverIp});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  CameraController? _controller;
  WebSocketChannel? _channel;
  bool _isStreaming = false;
  bool _wsReady = false;          // ← true only after WS handshake completes
  Timer? _timer;
  int _framesSent = 0;
  bool _isProcessingFrame = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _connectWebSocket();
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return;

      // Find rear camera
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      _controller = CameraController(
        camera,
        ResolutionPreset.medium, // 480p or 720p is good for CV
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _controller!.initialize();
      if (mounted) setState(() {});

      _startStreaming();
    } catch (e) {
      debugPrint('Camera error: $e');
    }
  }

  void _connectWebSocket() {
    try {
      // Use ws:// for local IPs, wss:// if using ngrok later
      final wsUrl = Uri.parse('ws://${widget.serverIp}:8000/ws/capture');
      _channel = WebSocketChannel.connect(wsUrl);

      // Listen for errors / disconnection so we know when the socket is dead
      _channel!.stream.listen(
        (_) {},   // server sends no replies on /ws/capture
        onError: (e) {
          debugPrint('WebSocket error: $e');
          if (mounted) setState(() => _wsReady = false);
        },
        onDone: () {
          debugPrint('WebSocket closed — will reconnect');
          if (mounted) setState(() => _wsReady = false);
          // Auto-reconnect after 1 second
          Future.delayed(const Duration(seconds: 1), () {
            if (mounted) _connectWebSocket();
          });
        },
      );

      // Perform the WebSocket handshake and mark as ready
      _channel!.ready.then((_) {
        debugPrint('WebSocket ready ✓');
        if (mounted) setState(() => _wsReady = true);
      }).catchError((e) {
        debugPrint('WebSocket handshake failed: $e');
        if (mounted) setState(() => _wsReady = false);
      });

      setState(() {});
    } catch (e) {
      debugPrint('WebSocket error: $e');
    }
  }

  void _startStreaming() {
    if (_controller == null || !_controller!.value.isInitialized) return;
    
    setState(() => _isStreaming = true);
    
    // Capture and send a frame every 200ms (5 FPS)
    _timer = Timer.periodic(const Duration(milliseconds: 200), (timer) async {
      if (!_isStreaming || _isProcessingFrame) return;
      
      try {
        _isProcessingFrame = true;
        // takePicture captures a JPEG on Android/iOS natively
        final XFile file = await _controller!.takePicture();
        final bytes = await file.readAsBytes();
        
        // Send binary JPEG to the server — only if WS handshake is complete
        if (_wsReady) _channel?.sink.add(bytes);
        
        // Delete the temporary file to prevent disk fill-up
        File(file.path).delete().catchError((_) {});
        
        if (mounted) {
          setState(() {
            _framesSent++;
          });
        }
      } catch (e) {
        debugPrint('Frame capture error: $e');
      } finally {
        _isProcessingFrame = false;
      }
    });
  }

  @override
  void dispose() {
    _isStreaming = false;
    _timer?.cancel();
    _controller?.dispose();
    _channel?.sink.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Capture Mode'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          Container(
            margin: const EdgeInsets.only(right: 16),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: _isStreaming ? Colors.red : Colors.grey,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  _isStreaming ? 'LIVE' : 'OFFLINE',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
                ),
              ],
            ),
          )
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Container(
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(24),
                color: Colors.black,
              ),
              clipBehavior: Clip.antiAlias,
              child: _controller != null && _controller!.value.isInitialized
                  ? CameraPreview(_controller!)
                  : const Center(child: CircularProgressIndicator()),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(24.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Status', style: TextStyle(color: Colors.white54)),
                    Text(
                      _wsReady ? 'Connected to Server' : 'Connecting...',
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    const Text('Frames Sent', style: TextStyle(color: Colors.white54)),
                    Text(
                      '$_framesSent',
                      style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.blue),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }
}
