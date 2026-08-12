import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class MonitorScreen extends StatefulWidget {
  final String serverIp;

  const MonitorScreen({super.key, required this.serverIp});

  @override
  State<MonitorScreen> createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen> {
  WebSocketChannel? _channel;
  Map<String, dynamic> _latestReport = {};
  bool _isConnected = false;

  @override
  void initState() {
    super.initState();
    _connectWebSocket();
  }

  void _connectWebSocket() {
    try {
      final wsUrl = Uri.parse('ws://${widget.serverIp}:8000/ws/monitor');
      _channel = WebSocketChannel.connect(wsUrl);
      
      setState(() => _isConnected = true);

      _channel!.stream.listen(
        (message) {
          if (mounted) {
            setState(() {
              _latestReport = jsonDecode(message);
            });
          }
        },
        onDone: () {
          if (mounted) setState(() => _isConnected = false);
        },
        onError: (error) {
          debugPrint('Monitor WS Error: $error');
          if (mounted) setState(() => _isConnected = false);
        },
      );
    } catch (e) {
      debugPrint('Connection error: $e');
    }
  }

  @override
  void dispose() {
    _channel?.sink.close();
    super.dispose();
  }

  // Map category strings to emojis and colors
  Map<String, dynamic> _getCategoryStyle(String category) {
    switch (category) {
      case 'Humans': return {'emoji': '👤', 'color': Colors.blue};
      case 'Vehicles': return {'emoji': '🚗', 'color': Colors.orange};
      case 'Animals': return {'emoji': '🐕', 'color': Colors.brown};
      case 'Electronics': return {'emoji': '💻', 'color': Colors.purple};
      case 'Furniture': return {'emoji': 's🪑', 'color': Colors.amber};
      default: return {'emoji': '📦', 'color': Colors.grey};
    }
  }

  @override
  Widget build(BuildContext context) {
    final List objects = _latestReport['objects'] ?? [];
    final double stability = _latestReport['scene_stability']?.toDouble() ?? 0.0;
    final String timestamp = _latestReport['timestamp'] ?? '--:--:--';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Monitor'),
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
                    color: _isConnected ? Colors.green : Colors.red,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  _isConnected ? 'CONNECTED' : 'OFFLINE',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
                ),
              ],
            ),
          )
        ],
      ),
      body: Column(
        children: [
          // Header Stats
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _buildStatBox('Objects', '${objects.length}', Colors.blue),
                _buildStatBox('Stability', '${stability.toInt()}%', Colors.green),
                _buildStatBox('Time', timestamp, Colors.purple),
              ],
            ),
          ),
          
          // Object List
          Expanded(
            child: objects.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.visibility_off, size: 64, color: Colors.white24),
                        const SizedBox(height: 16),
                        const Text(
                          'No objects detected',
                          style: TextStyle(color: Colors.white54, fontSize: 16),
                        ),
                      ],
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    itemCount: objects.length,
                    itemBuilder: (context, index) {
                      final obj = objects[index];
                      final style = _getCategoryStyle(obj['category'] ?? 'Other');
                      final isNew = obj['is_new'] == true;
                      final confidence = (obj['confidence'] ?? 0.0) * 100;

                      return Container(
                        margin: const EdgeInsets.only(bottom: 12),
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.surface,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: isNew ? Colors.blue.withOpacity(0.5) : Colors.white10,
                            width: isNew ? 1.5 : 1,
                          ),
                        ),
                        child: Column(
                          children: [
                            Row(
                              children: [
                                Text(style['emoji'], style: const TextStyle(fontSize: 24)),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Row(
                                        children: [
                                          Text(
                                            (obj['label'] as String).toUpperCase(),
                                            style: const TextStyle(
                                              fontWeight: FontWeight.bold,
                                              fontSize: 16,
                                            ),
                                          ),
                                          const SizedBox(width: 8),
                                          if (isNew)
                                            Container(
                                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                              decoration: BoxDecoration(
                                                color: Colors.blue,
                                                borderRadius: BorderRadius.circular(4),
                                              ),
                                              child: const Text('NEW', style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold)),
                                            ),
                                        ],
                                      ),
                                      Text(
                                        'ID #${obj['track_id']} • ${obj['duration']}',
                                        style: const TextStyle(color: Colors.white54, fontSize: 12),
                                      ),
                                    ],
                                  ),
                                ),
                                Text(
                                  '${confidence.toInt()}%',
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    color: style['color'],
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 12),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(2),
                              child: LinearProgressIndicator(
                                value: confidence / 100,
                                backgroundColor: Colors.white10,
                                valueColor: AlwaysStoppedAnimation<Color>(style['color']),
                                minHeight: 4,
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatBox(String label, String value, Color color) {
    return Container(
      width: 100,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        children: [
          Text(label, style: const TextStyle(color: Colors.white54, fontSize: 11)),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18, color: color),
          ),
        ],
      ),
    );
  }
}
