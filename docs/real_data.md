# การทดสอบกับสัญญาณจริง (ZCU216 raw IF)

ส่วน processing (block 8–17) เป็นโค้ดเดียวกับที่ใช้ใน simulation ข้อมูลจริงจึงผ่าน DDC, dechirp, range/Doppler และ calibration แบบเดียวกันทุกขั้น

```
ไฟล์ .bin + .yaml ──► load_capture ──► capture_to_frame ──► IQFrame ──► process() ──► กราฟ + moments.npz
                         (int16)        (DDC, หา chirp start)          (block 9–17)
```

## 1. Format ของไฟล์

ใช้ไฟล์ข้อมูลหนึ่งไฟล์คู่กับ YAML หนึ่งไฟล์ที่ชื่อเดียวกัน เช่น `run01.bin` + `run01.yaml`

### แบบ A: `adc_real` — raw IF จาก RF-ADC (แนะนำ)

- ตัวอย่างจริงแบบ real ช่องเดียว `int16` little-endian ที่ 2.5 GS/s ยังไม่ผ่าน DDC
- ถ้าข้อมูล 14 บิตอยู่ในบิตบนของ word 16 บิต (ค่า default ของ RF Data Converter) ให้ตั้ง `msb_aligned: true`
- 1 dwell ใน weather mode = 64 × 200 µs × 2.5 GS/s = 32 M samples = 64 MB

```yaml
format: adc_real
fs: 2.5e9
dtype: int16
byte_order: "<"
msb_aligned: true
adc_bits: 14
first_chirp_sample: null   # ถ้ารู้จาก MTS / trigger ให้ใส่ index ของ ADC sample ที่ chirp แรกเริ่ม
n_chirps: null             # null = ใช้ chirp ที่ครบทั้งหมด
notes: "run01, Pt 20 dBm, el 90 deg, rain"
# ใส่ key อื่นเพิ่มได้ (เช่น temperature, gain_setting) จะถูกเก็บไว้ใน frame.meta
```

### แบบ B: `ddc_iq` — หลัง DDC ในฮาร์ดแวร์

- I,Q สลับกันแบบ `int16` ที่อัตรา fs/decimation (เช่น 62.5 MS/s)
- ต้องใส่ `f_nco` ที่ใช้จริง และ `iq_full_scale` (ขนาด |I+jQ| ที่ได้เมื่อป้อน sine full scale เข้า ADC ให้วัดด้วย tone จริง เพราะ gain ของ DDC ในฮาร์ดแวร์ขึ้นกับการตั้งค่า mixer)

```yaml
format: ddc_iq
fs: 62.5e6
f_nco: 3.12432e9
iq_full_scale: 32768
iq_gain_db: 0.0
```

ถ้าต้องการไฟล์ตัวอย่างสำหรับทีม FPGA ให้รัน `scripts/make_test_capture.py` ไฟล์ `.yaml` ที่ได้คือ spec ของ format

### MIMO (4 RX พร้อมกัน)

- ใส่ `n_channels: 4` แล้ววางข้อมูลสลับกันทีละ sample: `ch0, ch1, ch2, ch3, ch0, ...` (ใช้ได้กับทั้ง `adc_real` และ `ddc_iq`; สำหรับ `ddc_iq` แต่ละ sample คือ I,Q)
- ใส่ `first_chirp_tx`: เลข TX (0–3) ของ chirp แรกที่ครบในไฟล์ ข้อมูลนี้ดูจากสัญญาณอย่างเดียวไม่ได้ ควรให้ FPGA เริ่ม capture ตรงกับ TX0 หรือบันทึกเลข TX ไว้ใน header
- config ต้องมี `mimo.enabled: true` และ chirp/TDM ตรงกับที่ส่งจริง
- calibrate ด้วย corner reflector ก่อน: `scripts/calibrate_mimo.py data/corner.bin --config ... --range 1.0 --angle 0` แล้วส่งไฟล์ `_cal.npy` ให้ `process_capture.py --cal`
- ตัวอย่าง: `scripts/make_test_capture.py --config config/lab_mimo.yaml --out data/test_lab_mimo`

## 2. Config ต้องตรงกับฮาร์ดแวร์

`process_capture.py` ใช้ config เพื่อสร้าง reference chirp และ calibration ดังนั้นค่าต่อไปนี้ต้องตรงกับที่ตั้งไว้ตอนเก็บข้อมูล:

| หมวด | ค่า |
|---|---|
| `lo` | `n_pll` (หรือ `f_pll`), `multiplier`, `sideband` |
| `waveform` | `f_rf_center`, `bandwidth`, `t_chirp`, `t_idle`, `n_chirps` (ต่อ dwell) |
| `adc` | `fs`, `bits`, `full_scale_dbm`, `ddc_decimation`, `nco_freq` |
| `frontend` | `pt_dbm`, gains, beamwidth, `rx_gain_db`, `lna` (ใช้ใน calibration → dBZ) |
| `processing` | `r_max`, `cal_offset_db`, `range_offset_m` (จากขั้นที่ 4) |

ควรทำ config แยกสำหรับฮาร์ดแวร์จริง เช่น `config/hw_weather.yaml` ที่มี `base: weather.yaml`

## 3. ประมวลผล

```bash
.venv/bin/python scripts/process_capture.py data/run01.bin --config config/hw_weather.yaml
```

ผลอยู่ใน `out/capture_run01/`:

- `00_capture_overview.png` แสดงสเปกตรัม raw ADC, peak dBFS, จำนวน sample ที่ clip, สเปกตรัมหลัง DDC และ correlation ที่ใช้หา chirp start
- `09`–`18` เป็นกราฟ block เดียวกับ simulation (ไม่มีเส้น truth)
- `summary.json`, `moments.npz` (range, dBZ, v, σv, SNR, valid, spectrum)

**การหา chirp start:** โค้ดทำ correlation กับ reference chirp (ใส่ Hann window ไว้เพื่อกด sidelobe) แล้วเลือก peak ที่มาเร็วที่สุด ซึ่งคือ leakage TX→RX ส่วน echo ที่แรงกว่าจะมาทีหลังเสมอ จากนั้นใช้ parabolic interpolation หาตำแหน่งระดับ sub-sample
ถ้าตั้ง Multi-Tile Sync และ trigger ระหว่าง DAC กับ ADC ได้ ให้ใส่ `first_chirp_sample` แทน วิธีนี้แม่นกว่าและไม่ต้องพึ่ง leakage

## 4. Calibration ด้วย corner reflector (ทำก่อนวัดฝน)

1. ใช้ config lab (B = 1 GHz) วาง trihedral ขนาดขอบที่รู้ค่าไว้ที่ระยะที่รู้ค่า (เช่น 1.00 m) ในที่โล่ง
2. เก็บ capture หนึ่งชุด แล้วรัน
   ```bash
   .venv/bin/python scripts/calibrate_corner.py data/corner.bin --config config/hw_lab.yaml --range 1.0 --edge 0.10
   ```
3. นำ `range_offset_m` และ `cal_offset_db` ที่ได้ไปใส่ใน config ของฮาร์ดแวร์ ค่านี้รวม gain/loss ของสาย, NF จริง และ mismatch ต่าง ๆ ไว้แล้ว

ถ้าลองกับ capture จำลอง (`data/test_lab.bin`) จะได้ offset 1.6 cm และ −0.24 dB ซึ่งใกล้ศูนย์ตามที่ควรเป็น

## 5. สิ่งที่ควรบันทึกเพิ่มในการวัดครั้งแรก

1. **Noise-only capture:** ปิด TX หรือต่อ load แทน antenna เพื่อวัด noise floor จริง แล้วเทียบกับ `noise_theory_dbm` เพื่อหา NF จริง
2. **Leakage-only capture:** หันเสาอากาศไปที่ท้องฟ้าใส ใช้วัด TX–RX isolation และดู headroom ของ ADC (peak dBFS)
3. **Loopback tone:** ป้อน tone ความถี่ที่รู้ค่า เพื่อยืนยัน sideband (USB/LSB) และ gain ของ DDC
4. **Corner reflector** ตามขั้นที่ 4
5. **ฝนจริง** พร้อมข้อมูลอ้างอิง (disdrometer หรือ rain gauge) สำหรับเทียบ Z–R

## 6. การแก้ปัญหา

| อาการ | สาเหตุที่เป็นไปได้ |
|---|---|
| peak อยู่ผิดระยะแบบคงที่ | ต้องใส่ `range_offset_m` หรือ chirp start ผิด (ดูกราฟ correlation ใน `00_capture_overview`) |
| range profile เป็นก้อนเลอะ ไม่มี peak | chirp ใน config ไม่ตรงกับฮาร์ดแวร์ หรือ sideband กลับด้าน (ลองกลับ slope) |
| peak dBFS ≈ 0, `clipped` > 0 | leakage แรงเกิน ต้องลด gain หน้า ADC หรือเพิ่ม isolation |
| noise สูงกว่าทฤษฎีมาก | NF จริงสูงกว่าที่คิด, spur จาก clock หรือ BPF ไม่ได้กันนอก zone |
| ความเร็วกลับเครื่องหมาย | sideband กลับด้าน หรือนิยาม I/Q สลับกัน |
