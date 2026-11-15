// DI-NET Android (M3): Compose UI поверх того же Rust-ядра.
// JNI-мост рукописный (крейт dinet-android), .so кладутся в jniLibs
// скриптом crates/dinet-android/build-android.sh — externalNativeBuild
// не используется.
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21" apply false
}
