# ── Flutter engine ───────────────────────────────────────────────────────────
-keep class io.flutter.** { *; }
-keep class io.flutter.embedding.** { *; }
-dontwarn io.flutter.**

# ── Camera plugin ─────────────────────────────────────────────────────────────
-keep class io.flutter.plugins.camera.** { *; }

# ── WebSocket / OkHttp (used by web_socket_channel) ──────────────────────────
-keep class okhttp3.** { *; }
-dontwarn okhttp3.**
-dontwarn okio.**

# ── Keep native JNI methods ───────────────────────────────────────────────────
-keepclassmembers class * {
    native <methods>;
}
