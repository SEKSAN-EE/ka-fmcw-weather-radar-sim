# Ka-band FMCW Weather Radar — Simulation Brief (ZCU216)

เอกสารนี้เป็น brief สำหรับเริ่มโปรเจกต์ใน Claude Code เป้าหมายคือสร้าง simulation แบบ end-to-end ของสายรับ (RX) ตั้งแต่สัญญาณสะท้อนจากฝนเข้าสู่ antenna ผ่าน down-converter, ADC ของ RFSoC ZCU216 ไปจนถึงการประมาณค่า weather moments (Z, v, σv) โดยแต่ละ block ต้องแสดง output ของตัวเองเป็นกราฟ เพื่อให้เห็นว่าสัญญาณเปลี่ยนไปอย่างไรในแต่ละขั้น

---

## 1. บริบทของระบบ (จาก block diagram)

ฮาร์ดแวร์เป็นแบบ SDR คือ DAC ของ FPGA สร้าง IF waveform เองทั้งหมด ระบบประกอบด้วย:

1. **TX IF chain (4 ช่อง):** DAC → BPF (custom) → IF amp (HMC311LP3) → 90° hybrid (RFHB02G08GVT) → IF_I / IF_Q ของ up-converter
2. **LO chain:** Ref clock 7.68 MHz จาก FPGA → PLL 8.54 GHz → splitter → LO amp (HMC311LP3) → 180° hybrid (RFHB04G18GPI) → LO-P / LO-N ของมิกเซอร์ทั้ง TX และ RX (LO ร่วมกัน → coherent)
3. **Up-converter:** ADMV1013 (24–44 GHz, LO quadrupler ภายใน ×4, รับ complex IF 0.8–6 GHz แบบ SSB)
4. **Array antenna:** TX 4 ช่อง, RX 4 ช่อง แยกกัน ไม่มี T/R switch (ส่ง-รับพร้อมกัน), gain ประมาณ 20 dBi, ทดสอบกับ corner reflector ที่ 1 m
5. **Down-converter:** ADMV1014 (LO ×4 ภายในเช่นเดียวกัน) → IF_I / IF_Q
6. **RX IF chain (4 ช่อง):** 90° hybrid รวม IF_I/IF_Q เป็น real IF (image-reject) → BPF (custom) → ADC ของ ZCU216

**ข้อสรุปสำคัญ:**

1. LO ที่มิกเซอร์จริง = 8.54 GHz × 4 = **34.16 GHz**
2. RF = 34.16 GHz ± IF (sideband ขึ้นกับการต่อ 0°/90° เข้า IF_I/IF_Q — ต้องยืนยันกับ datasheet หรือวัดจริง)
3. IF ฝั่งรับ = IF ฝั่งส่ง (LO ร่วมกัน) ถ้าต้องการ RF 37–38 GHz → IF = 2.84–3.84 GHz
4. ระบบนี้เป็น **FMCW แบบ digital dechirp**: ADC รับ chirp ที่ IF เต็มแถบ ไม่ใช่ beat signal การ dechirp ต้องทำใน FPGA/ซอฟต์แวร์

---

## 2. ข้อจำกัดของ ADC และแผนความถี่

ZCU216 (RFSoC Gen3): 16 × RF-ADC, 14-bit, สูงสุด 2.5 GSPS, real sampling, มี DDC (NCO + decimation) ในตัว

1. Bandwidth ต่อ Nyquist zone = fs/2 = 1.25 GHz (ไม่ใช่ 2.5 GHz)
2. IF 2.84–3.84 GHz คร่อมขอบ zone 3/4 (3.75 GHz) → เก็บเต็ม 1 GHz ไม่ได้ที่ fs ≤ 2.5 GSPS
3. **แผนที่เสนอ:** ปรับ PLL เป็น ≈ 8.594 GHz → LO×4 = 34.375 GHz → IF = 2.625–3.625 GHz อยู่กลาง zone 3 (2.5–3.75 GHz) มี guard ~125 MHz ทั้งสองข้าง, zone คี่ สเปกตรัมไม่กลับด้าน
4. สำหรับโหมดวัดฝน chirp bandwidth แคบ (หลัก MHz) ปัญหา zone แทบหายไป

Simulation ต้องทำให้ fs, LO, IF center, chirp bandwidth เป็นพารามิเตอร์ที่ปรับได้ และแสดงผลของ aliasing เมื่อวาง band ผิด zone

---

## 3. โหมดการทำงานที่ต้อง simulate

1. **Lab mode (validation):** B = 1 GHz, ΔR = 15 cm, corner reflector ที่ 1 m → ใช้ตรวจว่า pipeline ถูกต้อง
2. **Weather mode:** B ≈ 5–20 MHz (ΔR ≈ 7.5–30 m), chirp ยาว (เช่น 200 µs) เพื่อให้ delay ของเป้าไกล (5 km ≈ 33 µs) ยังเล็กกว่า chirp duration, เป้าหมายคือฝนแบบกระจาย (distributed target)

**Trade-off ที่ต้องแสดงใน simulation:** chirp ยาว → วัดไกลได้และ SNR ดีขึ้น แต่ v_max = λ/(4·T_rep) ลดลง (λ ≈ 8 mm, T_rep = 200 µs → v_max ≈ 10 m/s) ทำให้เกิด velocity aliasing

---

## 4. Processing pipeline (block ต่อ block)

แต่ละ block ต้องเป็นฟังก์ชันแยก มี input/output ชัดเจน และมีฟังก์ชัน plot ของตัวเอง

1. **Waveform generator (TX)** — สร้าง IF chirp ตามพารามิเตอร์ (f_start, B, T_chirp, T_idle, N_chirps) **Output:** สัญญาณ chirp ในโดเมนเวลา, spectrogram
2. **Scene / target model** — (ก) corner reflector: point target, RCS คงที่ (ข) ฝน: เม็ดฝนจำนวนมากกระจายใน range bin แต่ละช่อง, กำลังรวมจาก Z (dBZ) ผ่าน weather radar equation, ความเร็วแบบ Gaussian (mean v, width σv), ตำแหน่งอัปเดตทุก chirp, one-way/two-way attenuation ตาม rain rate (อาจใช้วิธี spectral time-series simulation แบบ Zrnić 1975 เป็นทางเลือกที่เร็วกว่า) **Output:** ground truth profile ของ Z(r), v(r), σv(r)
3. **Channel** — delay และ phase ของแต่ละ scatterer, Doppler shift, **TX→RX leakage** (antenna อยู่ติดกัน — สำคัญมากใน FMCW) **Output:** สัญญาณ RF-equivalent ที่ขา antenna RX
4. **Down-converter model (ADMV1014 + 90° hybrid)** — mix ด้วย LO×4, I/Q imbalance (gain/phase), image rejection ไม่สมบูรณ์ (~26 dBc), LO leakage **Output:** real IF, สเปกตรัมแสดง wanted sideband vs image
5. **Receiver noise** — thermal noise kT₀·B·F, NF เป็นพารามิเตอร์ (default 10 dB ไม่มี LNA; ทางเลือกใส่ LNA → คำนวณ NF ด้วยสูตร Friis) **Output:** IF + noise, SNR ที่ input ADC
6. **Anti-alias BPF** — bandpass ตาม Nyquist zone ที่เลือก **Output:** สเปกตรัมก่อน/หลังกรอง
7. **ADC (RF-ADC)** — sample ที่ fs, quantize 14-bit, clipping, (option) clock jitter **Output:** สเปกตรัมหลัง sampling แสดงการ fold ลง zone 1
8. **DDC** — NCO mix ที่ IF center → complex baseband, decimation filter (factor ปรับได้) **Output:** complex I/Q baseband, data rate ก่อน/หลัง decimate
9. **Digital dechirp** — คูณด้วย conjugate ของ reference chirp (ต้องชดเชยให้ reference ตรงกับที่ผ่าน DDC แล้ว) **Output:** beat signal; ความถี่ beat f_b = S·τ
10. **Low-pass + decimate ครั้งที่สอง** — เหลือเฉพาะช่วง beat ที่สนใจ (หลัก MHz) **Output:** beat signal อัตราต่ำ
11. **Range FFT** — window (Hann/Blackman), zero-padding option **Output:** range profile ต่อ chirp → range–chirp matrix
12. **Leakage / clutter removal** — ลบ TX leakage ใน near range, zero-Doppler notch หรือ mean subtraction ตามแกน chirp **Output:** range profile ก่อน/หลัง
13. **Doppler processing ต่อ range bin** — (ก) pulse-pair: R(0), R(1) → power, v = −(λ/4πT)·arg R(1), σv จาก |R(1)|/R(0) (ข) Doppler FFT → Doppler spectrum ต่อ bin **Output:** range–Doppler map, moments ต่อ bin
14. **Noise estimation + thresholding** — ประมาณ noise floor, ตัดทิ้ง bin ที่ SNR ต่ำกว่า threshold, noise-corrected power **Output:** mask ของ bin ที่ valid
15. **Calibration → reflectivity** — radar constant (Pt, G, beamwidth, λ, |K|², ΔR, losses), range correction r² → Z (dBZ) **Output:** Z(r) เทียบกับ ground truth
16. **Attenuation correction (optional)** — แก้ Ka-band attenuation แบบ iterative (Hitschfeld–Bordan) **Output:** Z ก่อน/หลังแก้
17. **Rain rate** — Z–R relation (Z = aR^b, default Marshall–Palmer a=200, b=1.6) **Output:** R(r) มม./ชม.
18. **(Future) MIMO angle** — 4TX × 4RX virtual array → angle FFT **Output:** range–angle map

---

## 5. Validation ที่ต้องมี

1. Lab mode: peak ของ corner reflector ต้องอยู่ที่ 1.00 m ± 1 bin และ SNR สอดคล้องกับ radar equation แบบ point target
2. Weather mode: Z, v, σv ที่ประมาณได้ต้องใกล้ ground truth (plot error เทียบกับ SNR)
3. กราฟ SNR vs range ของฝน 0/20/35/45 dBZ เทียบกับ link budget ที่คำนวณไว้ (Pt 0 และ 20 dBm, NF 10 dB, ΔR 30 m, T_coh 1 ms) — ฝน 20–35 dBZ ควรวัดได้ประมาณ 1–3 km ที่ Pt 20 dBm
4. แสดง aliasing เมื่อ IF คร่อมขอบ Nyquist zone เทียบกับแผน LO 8.594 GHz
5. แสดงผลของ velocity aliasing และทดลอง unfolding

---

## 6. พารามิเตอร์ default (ต้องปรับได้ทั้งหมดจาก config ไฟล์เดียว)

1. f_ref = 7.68 MHz, f_PLL = 8.54 GHz (ทางเลือก 8.594 GHz), LO multiplier = 4
2. RF ≈ 37–38 GHz, λ ≈ 8 mm
3. fs_ADC = 2.5 GSPS, 14-bit, DDC decimation ปรับได้
4. Pt = 0 dBm (ไม่มี PA) / 20 dBm (มี PA), G = 20 dBi, beamwidth ≈ 20° × 20°
5. NF = 10 dB (ไม่มี LNA) / ≈ 3.2 dB (มี LNA NF 3 dB, gain 20 dB)
6. |K|² = 0.88
7. Lab: B = 1 GHz; Weather: B = 10 MHz, T_chirp = 200 µs, N_chirps = 64
8. ค่า NF, Pt, sideband, ค่า IF จริงเป็น**สมมติฐาน** ต้องแทนด้วยค่าจากฮาร์ดแวร์เมื่อทราบ

---

## 7. ข้อควรระวังด้าน computation

การ simulate ที่ IF 2.5 GSPS ตลอด chirp 200 µs × 64 chirps ใช้ sample หลายสิบล้านต่อช่อง ให้ทำแบบนี้:

1. Simulate เต็มสาย IF → ADC → DDC เพียงไม่กี่ chirp เพื่อแสดงและตรวจ block 4–8
2. เมื่อยืนยันแล้วว่าให้ผลเท่ากัน ใช้ **complex baseband equivalent model** สำหรับ chirp ที่เหลือและสำหรับ weather mode
3. เขียนเทสต์ที่ยืนยันว่าสองวิธีให้ผลตรงกัน

---

## 8. โครงสร้างโปรเจกต์ที่ต้องการ

1. ภาษา: Python (numpy, scipy, matplotlib), config เป็น YAML หรือ dataclass
2. โครงสร้าง: `config/`, `radar_sim/` (หนึ่งโมดูลต่อหนึ่งกลุ่ม block), `notebooks/` (walkthrough ทีละ block พร้อมกราฟ), `tests/`
3. มี script เดียวที่รันทั้ง pipeline แล้วบันทึกกราฟทุก block ลงโฟลเดอร์ output
4. ออกแบบให้ block 8–17 รับ input จากไฟล์ข้อมูลจริงของ ZCU216 ได้ในอนาคต (แยก simulation กับ processing ออกจากกัน)

---

## 9. คำถามที่ยังเปิดอยู่ (ต้องยืนยันกับฮาร์ดแวร์)

1. IF จริงที่ DAC ส่ง และ sideband ที่ ADMV1013 เลือก (USB/LSB)
2. Pt จริงที่ขา antenna และมี PA / LNA หรือไม่
3. NF จริงของ ADMV1014 ในการตั้งค่าที่ใช้
4. Isolation ระหว่าง antenna TX กับ RX (กำหนดระดับ leakage)
5. Beam pattern จริงของ array และมุมที่ใช้ยิง (แนวตั้ง/แนวนอน)
6. โหมดการใช้งาน MIMO (TDM หรือ simultaneous)

---

## 10. ลำดับการทำงานที่แนะนำ

1. ตั้งโครงสร้างโปรเจกต์และ config
2. Block 1, 9–11 ใน baseband ก่อน กับ corner reflector (ตรวจ range ได้เร็วที่สุด)
3. เพิ่ม block 4–8 (IF, ADC, DDC) และเทสต์เทียบกับ baseband
4. เพิ่ม weather target model (block 2) และ block 12–15
5. Validation ตามข้อ 5
6. Attenuation correction, rain rate, MIMO ตามลำดับ
