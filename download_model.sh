#!/usr/bin/env bash
# Скачивает замороженную rubert-tiny2 (веса и токенизатор) с Hugging Face на
# закреплённой ревизии и проверяет SHA-256 каждого файла.
#
# Использование: ./download_model.sh [папка]   (по умолчанию starter/assets/model)
# Зеркало можно задать через HF_ENDPOINT, например HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail

REPO="cointegrated/rubert-tiny2"
REVISION="e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
DEST="${1:-starter/assets/model}"

FILES=(
  "config.json adee8b3e344bcb8379f44d0b3577c267d52881341e05d973c43e49974778dfff"
  "special_tokens_map.json 303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"
  "tokenizer_config.json 74aab51b71a8d116c035464df96c600770f9844696b8965e397b2b1649010686"
  "tokenizer.json 45cc9f974145661db6bc020795839d1dc371adc19a9c78b910393209b4fe5efc"
  "vocab.txt f056a69b097422652053bf87565c35543e5d81540ca4b7dddd28de4157a969e0"
  "model.safetensors 26ebb6db2a68593c54c74902d7a74f332da66297693f965cc9f1b0af4abf3894"
)

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

mkdir -p "$DEST"
for entry in "${FILES[@]}"; do
  read -r name expected <<<"$entry"
  target="$DEST/$name"
  if [[ -f "$target" && "$(sha256 "$target")" == "$expected" ]]; then
    echo "уже есть  $name"
    continue
  fi
  echo "загрузка  $name"
  curl -fL --retry 3 --progress-bar -o "$target.part" \
    "$ENDPOINT/$REPO/resolve/$REVISION/$name"
  actual="$(sha256 "$target.part")"
  if [[ "$actual" != "$expected" ]]; then
    rm -f "$target.part"
    echo "ошибка: SHA-256 не совпадает для $name (получено $actual)" >&2
    exit 1
  fi
  mv "$target.part" "$target"
done
echo "Модель готова: $DEST"
