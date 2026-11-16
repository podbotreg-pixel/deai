#!/usr/bin/env bash
# Кросс-сборка телефонного ядра DI-NET (M3): libdinet_phone.so для Android.
#
# Без cargo-ndk и без gradle externalNativeBuild: cargo умеет собирать
# android-таргеты сам, если дать ему линкер из NDK. minSdk 24 → clang с
# суффиксом API (aarch64-linux-android24-clang, без дефиса перед API).
# Результат копируется в jniLibs Android-проекта:
#   arm64-v8a/   — реальный телефон;
#   x86_64/      — эмулятор (AVD, живая проверка шага 1).
#
# Требует: rustup target add aarch64-linux-android x86_64-linux-android
# Учет:    ANDROID_NDK_HOME переопределяет путь к NDK.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"

NDK="${ANDROID_NDK_HOME:-$HOME/Android/Sdk/ndk/27.2.12479018}"
API=24
TC="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/android/app/src/main/jniLibs"
mkdir -p "$OUT"

[ -x "$TC/aarch64-linux-android${API}-clang" ] || {
  echo "NDK clang не найден: $TC" >&2
  exit 1
}

for pair in "aarch64-linux-android:arm64-v8a" "x86_64-linux-android:x86_64"; do
  tgt="${pair%%:*}"
  dir="${pair##*:}"
  up="$(echo "$tgt" | tr 'a-z-' 'A-Z_')"
  export "CARGO_TARGET_${up}_LINKER=$TC/${tgt}${API}-clang"
  # ring (libp2p-noise) собирает C через cc-rs: для android-таргетов cc-rs
  # ищет "{triple}-clang" (БЕЗ API-суффикса) на PATH, затем CC_<target>.
  # NDK 27 держит только API-суффиксные обёртки — делаем шимы-симлинки и
  # ставим bin NDK первым в PATH.
  [ -e "$TC/$tgt-clang" ] || ln -s "${tgt}${API}-clang" "$TC/$tgt-clang"
  export "CC_${up}=$TC/${tgt}${API}-clang"
  export "AR_${up}=$TC/llvm-ar"
  export PATH="$TC:$PATH"
  cargo build --release -p dinet-android --target "$tgt"
  mkdir -p "$OUT/$dir"
  cp -f "$ROOT/target/$tgt/release/libdinet_phone.so" "$OUT/$dir/libdinet_phone.so"
  echo "jniLibs/$dir/libdinet_phone.so: $(du -h "$OUT/$dir/libdinet_phone.so" | cut -f1)"
done
