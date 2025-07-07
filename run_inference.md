```sh
uv run inference.py \
    --type tfgridnet \
    --mix ./test_data/normal_cocktail_party_test_data.wav \
    --ref ./test_data/nana_speaker_vocals.wav \
    --out ./test_data/output.wav
```

```sh
uv run inference.py \
    --type tfgridnet \
    --mix ./test_data/normal_cocktail_party_test_data.wav \
    --ref ./test_data/nana_speaker_vocals.wav \
    --out ./test_data/output.wav
```


```sh
uv run modal run modal_entry.py \
    --type tfgridnet \
    --mix /root/USEF-TSE/normal_cocktail_party_test_data.wav \
    --ref /root/USEF-TSE/nana_speaker_vocals.wav \
    --out /output/output_tfgridnet.wav
```

```sh
uv run modal run modal_entry.py \
    --type sepformer \
    --mix /root/USEF-TSE/normal_cocktail_party_test_data.wav \
    --ref /root/USEF-TSE/nana_speaker_vocals.wav \
    --out /output/output_sepformer.wav
```