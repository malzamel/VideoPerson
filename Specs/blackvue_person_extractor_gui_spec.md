# مواصفات برنامج BlackVue Person Extractor — نسخة GUI

## الهدف العام

بناء برنامج يعمل على Windows بواجهة رسومية سهلة، يأخذ بطاقة الذاكرة الخاصة بكاميرا **BlackVue DR970X LTE Plus**، ينسخ ملفات الفيديو من الـSD إلى الجهاز، يفهرسها، يحللها، ثم يستخرج صور الأشخاص/الوجوه التي ظهرت في الفيديو، مع اختيار أوضح صورة ممكنة لكل ظهور، وإمكانية حفظ مقطع فيديو قصير حول وقت ظهور الشخص.

هذه الوثيقة مخصصة لإعطائها إلى Cursor ليبني البرنامج.

---

## نص التكليف لـ Cursor

```text
You are a senior Python desktop application engineer.

Build a Windows GUI application for processing BlackVue DR970X LTE Plus dashcam footage.

Project name:
blackvue-person-extractor-gui

Main goal:
Create a local Windows desktop application with a simple GUI that:
1. Detects/selects a BlackVue SD card or source folder.
2. Copies video files safely from the SD card to a local archive on the computer.
3. Indexes all imported video files in SQLite.
4. Processes the videos locally.
5. Detects people and faces in the videos.
6. Selects the clearest face image for each appearance/person.
7. Exports best face images, optional person crops, short video clips, and a local HTML/CSV report.
8. Does not upload anything to the cloud.
9. Does not identify people by name.
10. Does not compare faces against external databases.

Important context:
- Camera model: BlackVue DR970X LTE Plus.
- It records many short MP4 files, typically 1-minute files, not one long video.
- It may be 2CH:
  - Front camera: 4K, 3840x2160, 30fps.
  - Rear camera: Full HD, 1920x1080, 30fps.
- Codec may be H.265/HEVC or H.264/AVC.
- Files can be large.
- The program must never process directly from the SD card.
- The program must copy files to the computer first.
- The program must never modify or delete files on the SD card.

BlackVue filename pattern:
YYYYMMDD_HHMMSS_[Recording Type][Recording Direction][Other].mp4

Examples:
20260517_143000_NF.mp4
20260517_143000_NR.mp4
20260517_143000_EF.mp4
20260517_143000_IR.mp4

Meaning:
N = Normal recording
E = Driving impact/event
I = Parking impact
P = Parking motion / parking recording
M = Manual
F = Front camera
R = Rear camera

The application must be designed around many short video segments.
Do not assume a single long video.

Target platform:
- Windows 11
- Python 3.11 or 3.12
- Local-only processing
- GUI-first
- No cloud upload
- No external API calls
- No online face recognition

Recommended stack:
- Python
- PySide6 for GUI
- SQLite for database
- FFmpeg / ffprobe for metadata and clip export
- OpenCV for frame reading
- Ultralytics YOLO for person detection and tracking
- MediaPipe Face Detector for face detection
- NumPy
- Pillow
- Rich/logging for internal logs
- Optional later: ONNX Runtime for optimized inference
- Optional later: InsightFace for local-only face clustering, disabled by default

Important development approach:
Build a GUI-first application, but keep the video processing logic in separate backend modules.
The GUI must call backend services and must not contain the heavy processing logic directly.
The UI should remain responsive during import and processing.
Use worker threads or background tasks inside the app.

Do not start with a CLI-only app.
A small internal CLI for debugging is acceptable, but the main product must be a GUI.
```

---

# 1. واجهة البرنامج المطلوبة

## الشاشة الأولى: اختيار المصدر والوجهة

اسم الشاشة:
**Import Videos**

المكونات:

- حقل اختيار مصدر الفيديو:
  - زر: `Select SD Card / Source Folder`
  - يسمح باختيار حرف الدرايف مثل `E:\` أو مجلد فيه ملفات BlackVue.

- حقل اختيار مجلد الحفظ:
  - زر: `Select Archive Folder`
  - مثال: `D:\BlackVueArchive`

- اسم الحالة / المشروع:
  - حقل نصي: `Case Name`
  - مثال: `2026-05-17`

- خيارات الاستيراد:
  - Checkbox: `Verify copied files by size` — مفعّل افتراضياً.
  - Checkbox: `Calculate SHA256 hash after copy` — غير مفعّل افتراضياً لأنه أبطأ.
  - Checkbox: `Skip files already copied` — مفعّل افتراضياً.

- زر:
  - `Scan Source`

- بعد الفحص، يعرض البرنامج:
  - عدد ملفات MP4 الموجودة.
  - عدد ملفات BlackVue المعروفة.
  - عدد ملفات Front.
  - عدد ملفات Rear.
  - الحجم الإجمالي المتوقع.
  - تحذير إذا الوجهة لا تحتوي مساحة كافية.

- زر:
  - `Import to Computer`

المتطلبات:
- لا يبدأ التحليل قبل نسخ الملفات إلى الجهاز.
- لا يتم تعديل الـSD.
- لا يتم حذف أي ملف من الـSD.
- لو انقطع البرنامج، يمكن استكمال الاستيراد لاحقاً.

---

## الشاشة الثانية: حالة الاستيراد

اسم الشاشة:
**Import Progress**

تعرض:

- Progress bar عام.
- Progress bar للملف الحالي.
- اسم الملف الجاري نسخه.
- عدد الملفات المنسوخة.
- عدد الملفات المتخطاة.
- عدد الملفات التي فشل نسخها.
- الحجم المنسوخ.
- سجل مختصر للأخطاء.

أزرار:
- `Pause`
- `Resume`
- `Cancel`
- `Open Archive Folder`

المتطلبات:
- النسخ يكون على دفعات chunks مثل 16MB أو 64MB.
- إذا الملف موجود مسبقاً والحجم مطابق، يتم تخطيه.
- إذا الملف موجود والحجم مختلف، يحفظ باسم conflict أو يسجل كـ conflict ولا يستبدله بصمت.
- يتم تسجيل كل عملية في SQLite.

---

## الشاشة الثالثة: إعدادات التحليل

اسم الشاشة:
**Processing Settings**

بعد الاستيراد، تعرض هذه الشاشة.

الخيارات:

### اختيار الكاميرا
- Radio buttons:
  - `Front only`
  - `Rear only`
  - `Both` — افتراضي.

### نوع التسجيل
Checkboxes:
- `Normal`
- `Driving Impact`
- `Parking Impact`
- `Parking Motion / Parking`
- `Manual`
- `Overspeed`
- `Other`

افتراضياً:
- Normal + Driving Impact + Parking Impact + Manual.

### سرعة أخذ العينات
حقل:
- `Sample FPS`

القيم:
- افتراضي: 3 fps
- للأحداث: يمكن 5 fps
- لا تحلل 30fps كاملة افتراضياً.

### إعدادات التعرف
- Person confidence threshold:
  - default: 0.35
- Face confidence threshold:
  - default: 0.50
- Minimum face width:
  - default: 40 px
  - recommended: 80 px for useful output.

### إعدادات الحفظ
Checkboxes:
- `Save best face image` — مفعّل.
- `Save best person crop` — مفعّل.
- `Save top face candidates` — مفعّل.
- `Generate short video clip` — مفعّل.
- `Generate contact sheet` — مفعّل.
- `Generate HTML report` — مفعّل.
- `Generate CSV report` — مفعّل.

### مدة مقطع الفيديو
- Seconds before appearance:
  - default: 3 seconds.
- Seconds after appearance:
  - default: 3 seconds.

زر:
- `Start Processing`

---

## الشاشة الرابعة: حالة التحليل

اسم الشاشة:
**Processing Progress**

تعرض:

- إجمالي الملفات.
- الملفات المعالجة.
- الملفات المتبقية.
- الملف الحالي.
- الكاميرا الحالية: Front / Rear.
- عدد الأشخاص المكتشفين.
- عدد الوجوه المقبولة.
- عدد الوجوه المرفوضة بسبب الجودة.
- آخر صورة وجه مرشحة، thumbnail صغير.
- Progress bar عام.
- سجل مختصر للأحداث.

أزرار:
- `Pause`
- `Resume`
- `Stop after current file`
- `Open Logs`

المتطلبات:
- يجب ألا تتجمد الواجهة أثناء التحليل.
- يجب تشغيل المعالجة في Worker Thread أو QThread.
- يجب تحديث الواجهة عن طريق signals/slots.
- إذا توقف البرنامج، يمكن استئناف المعالجة.
- الملفات التي انتهت لا تعاد معالجتها إلا إذا اختار المستخدم ذلك.

---

## الشاشة الخامسة: النتائج

اسم الشاشة:
**Results Gallery**

تعرض النتائج على شكل بطاقات.

كل بطاقة لشخص/ظهور تحتوي:
- أفضل صورة وجه.
- وقت الظهور.
- مدة الظهور.
- الكاميرا: Front / Rear.
- نوع التسجيل: Normal / Event / Parking / Manual.
- درجة جودة الوجه.
- زر: `Open Clip`
- زر: `Open Folder`
- زر: `Show Candidates`
- زر: `Export Selected`

خيارات الفرز:
- Sort by time.
- Sort by best face score.
- Sort by camera.
- Sort by recording type.

خيارات الفلترة:
- Front / Rear.
- Minimum quality score.
- Recording type.
- Date/time range.

أزرار عامة:
- `Open Output Folder`
- `Export Summary CSV`
- `Open HTML Report`
- `Reprocess Selected`
- `Delete Output for Selected` — يحذف فقط الملفات الناتجة، وليس الأصلية.

---

# 2. هيكل المشروع

```text
blackvue-person-extractor-gui/
  pyproject.toml
  README.md
  .gitignore

  models/
    README.md

  data/
    archive/
    output/
    db/
    logs/

  src/
    blackvue_person_extractor/
      __init__.py

      app.py
      config.py
      logging_config.py

      gui/
        __init__.py
        main_window.py
        import_page.py
        import_progress_page.py
        processing_settings_page.py
        processing_progress_page.py
        results_page.py
        widgets.py
        workers.py

      core/
        __init__.py
        blackvue_filename.py
        sd_scanner.py
        importer.py
        ffmpeg_utils.py
        video_indexer.py
        video_reader.py
        person_detector.py
        face_detector.py
        quality.py
        processor.py
        clip_exporter.py
        contact_sheet.py
        reporting.py

      storage/
        __init__.py
        db.py
        models.py
        repositories.py

      utils/
        __init__.py
        paths.py
        hashing.py
        image_utils.py
        time_utils.py

  tests/
    test_blackvue_filename.py
    test_importer.py
    test_quality.py
    test_db.py
```

---

# 3. قاعدة البيانات SQLite

اسم قاعدة البيانات:
`blackvue_person_extractor.sqlite`

مكانها:
```text
D:\BlackVueArchive\<case-name>\db\blackvue_person_extractor.sqlite
```

## جدول cases

```text
cases
- id INTEGER PRIMARY KEY
- case_name TEXT
- archive_path TEXT
- created_at TEXT
- notes TEXT
```

## جدول video_files

```text
video_files
- id INTEGER PRIMARY KEY
- case_id INTEGER
- original_path TEXT
- archive_path TEXT
- filename TEXT
- start_datetime TEXT
- end_datetime TEXT
- recording_type_code TEXT
- recording_type_label TEXT
- camera_direction_code TEXT
- camera_direction_label TEXT
- size_bytes INTEGER
- sha256 TEXT
- width INTEGER
- height INTEGER
- fps REAL
- duration_seconds REAL
- codec TEXT
- imported_at TEXT
- indexed_at TEXT
- import_status TEXT
- processing_status TEXT
- error_message TEXT
```

قيم `processing_status`:
```text
pending
processing
processed
failed
skipped
```

## جدول appearance_tracks

```text
appearance_tracks
- id INTEGER PRIMARY KEY
- case_id INTEGER
- first_file_id INTEGER
- last_file_id INTEGER
- camera_direction_code TEXT
- recording_type_code TEXT
- local_track_id TEXT
- global_track_id TEXT
- start_datetime TEXT
- end_datetime TEXT
- start_ms INTEGER
- end_ms INTEGER
- best_frame_file_id INTEGER
- best_frame_ms INTEGER
- best_face_score REAL
- best_face_path TEXT
- best_person_crop_path TEXT
- clip_path TEXT
- contact_sheet_path TEXT
- created_at TEXT
```

## جدول face_candidates

```text
face_candidates
- id INTEGER PRIMARY KEY
- appearance_track_id INTEGER
- file_id INTEGER
- frame_ms INTEGER
- face_confidence REAL
- person_confidence REAL
- face_bbox_json TEXT
- person_bbox_json TEXT
- sharpness_score REAL
- brightness_score REAL
- face_size_score REAL
- frontality_score REAL
- total_quality_score REAL
- image_path TEXT
```

## جدول processing_runs

```text
processing_runs
- id INTEGER PRIMARY KEY
- case_id INTEGER
- started_at TEXT
- finished_at TEXT
- status TEXT
- settings_json TEXT
- error_message TEXT
```

---

# 4. تحليل أسماء ملفات BlackVue

ملف:
`core/blackvue_filename.py`

المطلوب:
- قراءة أسماء الملفات.
- استخراج التاريخ والوقت.
- استخراج نوع التسجيل.
- استخراج اتجاه الكاميرا.
- عدم التعطل إذا الاسم غير مطابق.

Regex مقترح:
```text
^(\d{8})_(\d{6})_([A-Z])([FR])([A-Z0-9]*)?\.mp4$
```

المخرجات:
```text
recording_date
recording_time
start_datetime
recording_type_code
recording_type_label
camera_direction_code
camera_direction_label
other_code
extension
is_valid_blackvue_name
```

خريطة نوع التسجيل:
```text
N = Normal
P = Parking Motion / Parking Time-lapse
M = Manual
E = Driving Impact
I = Parking Impact
O = Overspeed
A = Acceleration
T = Hard Cornering
B = Hard Braking
R = Geofence Enter
X = Geofence Exit
G = Geofence Pass
D = Drowsiness
L = Distraction
Y = Seatbelt
F = Undetected
```

خريطة الكاميرا:
```text
F = Front
R = Rear
```

---

# 5. فحص الـSD والاستيراد

ملفات:
```text
core/sd_scanner.py
core/importer.py
```

المتطلبات:

- يختار المستخدم مصدر الفيديو من الواجهة.
- يفحص البرنامج المصدر recursively.
- يبحث عن ملفات `.mp4`.
- يعطي أولوية للمجلدات التي قد تحتوي BlackVue/Record/Event/Parking، لكن لا يعتمد عليها فقط.
- يعرض ملخصاً قبل النسخ.
- لا يحلل من الـSD.
- ينسخ كل الملفات إلى الجهاز أولاً.

هيكل الحفظ:

```text
D:\BlackVueArchive\
  2026-05-17\
    original\
      20260517_143000_NF.mp4
      20260517_143000_NR.mp4
    db\
      blackvue_person_extractor.sqlite
    logs\
      import.log
      process.log
    output\
```

منطق النسخ:
- Chunked copy.
- Chunk size: 64MB.
- إظهار التقدم.
- لو الملف موجود والحجم مطابق: skip.
- لو الملف موجود والحجم مختلف: conflict.
- تحقق بالحجم افتراضياً.
- SHA256 اختياري.
- الاستيراد قابل للاستكمال.

---

# 6. فهرسة الفيديو

ملفات:
```text
core/ffmpeg_utils.py
core/video_indexer.py
```

استخدم ffprobe لاستخراج:
```text
duration
width
height
fps
codec
streams
```

المتطلبات:
- لا تحمل الفيديو كاملاً في الذاكرة.
- إذا فشل ffprobe، سجّل الخطأ وانتقل للملف التالي.
- احسب `end_datetime = start_datetime + duration`.
- اربط ملفات الأمامي والخلفي لنفس الدقيقة عند تطابق timestamp.

---

# 7. قراءة الإطارات

ملف:
```text
core/video_reader.py
```

المتطلبات:
- لا تحلل كل 30 إطاراً في الثانية افتراضياً.
- sample_fps الافتراضي = 3.
- يمكن رفعه للأحداث إلى 5.
- اقرأ frame by frame.
- لا تحمل الفيديو كاملاً.
- عند 4K، يمكن تصغير نسخة للكشف مثلاً إلى عرض 1280 أو 1920.
- احتفظ بإحداثيات يمكن تحويلها إلى الإطار الأصلي.
- استخدم OpenCV أولاً.
- إذا فشل مع H.265، استخدم FFmpeg fallback.
- لو فشل ملف، سجّل الخطأ وأكمل.

---

# 8. اكتشاف الأشخاص

ملف:
```text
core/person_detector.py
```

استخدم:
```text
Ultralytics YOLO
```

المطلوب:
- اكتشاف class = person فقط.
- model قابل للتغيير من الإعدادات.
- افتراضي:
  - yolo11n.pt أو yolo11s.pt حسب المتاح.
- confidence default = 0.35.
- IoU default = 0.5.

التتبع:
- استخدم ByteTrack افتراضياً.
- BoT-SORT خيار لاحق.
- أعط كل ظهور local_track_id داخل الملف.
- في النسخة الأولى، مسموح أن يكون التتبع داخل كل ملف فقط.
- لاحقاً نربط الظهور بين ملفين متتاليين لأن ملفات BlackVue دقيقة واحدة.

مهم:
إذا ظهر شخص في آخر ثواني من ملف واستمر في بداية الملف التالي، صمم قاعدة البيانات بحيث يمكن دمج الظهور لاحقاً.

---

# 9. اكتشاف الوجوه داخل الأشخاص

ملف:
```text
core/face_detector.py
```

استخدم:
```text
MediaPipe Face Detector
```

المنطق:
1. بعد اكتشاف الشخص، قص منطقة الشخص من الإطار الأصلي.
2. شغّل اكتشاف الوجه داخل قصاصة الشخص.
3. إذا وجد وجه:
   - حوّل bbox إلى إحداثيات الإطار الأصلي.
   - قيّم جودة الوجه.
   - احفظ المرشح إذا اجتاز الحد الأدنى.
4. إذا لم يوجد وجه:
   - احتفظ بظهور الشخص، لكن لا تحفظ صورة وجه.

المعايير الدنيا:
```text
face_confidence >= 0.50
face_width >= 40 px
sharpness acceptable
brightness acceptable
```

ممنوع:
- لا تحاول معرفة اسم الشخص.
- لا تربط الوجه بقواعد بيانات خارجية.
- لا تستخدم خدمات سحابية.

---

# 10. اختيار أوضح وجه

ملف:
```text
core/quality.py
```

الهدف:
اختيار أفضل صورة ممكنة للوجه لكل ظهور.

معايير الجودة:
1. ثقة اكتشاف الوجه.
2. حجم الوجه.
3. حدة الصورة / عدم الاهتزاز.
4. الإضاءة.
5. اتجاه الوجه / frontality.
6. اكتمال الوجه داخل الإطار.

معادلة مقترحة:
```text
total_quality_score =
  0.25 * face_confidence_score +
  0.25 * face_size_score +
  0.25 * sharpness_score +
  0.15 * brightness_score +
  0.10 * frontality_score
```

Sharpness:
- استخدم variance of Laplacian على grayscale.

Brightness:
- متوسط الإضاءة.
- أعلى درجة حين يكون المتوسط تقريباً بين 100 و180.

Face size:
- كلما كان الوجه أكبر كان أفضل إلى حد معين.

Frontality:
- إذا توفرت landmarks:
  - تماثل العينين.
  - الأنف قريب من المنتصف.
  - الفم قريب من المنتصف.
- إذا لم تتوفر، أعط score متوسط.

المخرجات لكل ظهور:
```text
best_face.jpg
best_person_crop.jpg
contact_sheet.jpg
top_candidates/
metadata.json
```

---

# 11. استخراج مقاطع الفيديو

ملف:
```text
core/clip_exporter.py
```

المطلوب:
- لكل ظهور، احفظ مقطع فيديو قصير حوله.
- افتراضياً:
  - 3 ثواني قبل بداية الظهور.
  - 3 ثواني بعد نهاية الظهور.

التحدي:
ملفات BlackVue دقيقة واحدة.
قد يبدأ الظهور في آخر الملف ويكمل في الملف التالي.

المطلوب:
- إذا المقطع داخل ملف واحد، قص منه مباشرة.
- إذا يتجاوز حدود الملف:
  - استخدم الملف السابق/الحالي/التالي حسب الحاجة.
  - ادمج المقاطع أو أخرج clip واحد.
- استخدم FFmpeg.
- حاول stream copy أولاً:
```text
-c copy
```
- إذا فشل أو كان القطع غير دقيق، استخدم re-encode fallback.

---

# 12. مخرجات البرنامج

هيكل المخرجات:

```text
D:\BlackVueArchive\
  2026-05-17\
    original\
      20260517_143000_NF.mp4
      20260517_143000_NR.mp4

    db\
      blackvue_person_extractor.sqlite

    logs\
      import.log
      process.log

    output\
      summary.html
      summary.csv
      persons\
        appearance_000001\
          best_face.jpg
          best_person_crop.jpg
          contact_sheet.jpg
          clip.mp4
          metadata.json

        appearance_000002\
          best_face.jpg
          best_person_crop.jpg
          contact_sheet.jpg
          clip.mp4
          metadata.json
```

ملف `metadata.json` يحتوي:
```text
appearance_id
start_datetime
end_datetime
source_files
camera_direction
recording_type
best_face_score
best_frame_timestamp
best_face_path
clip_path
```

ملف `summary.csv` يحتوي:
```text
appearance_id
start_datetime
end_datetime
camera
recording_type
best_face_score
best_face_path
clip_path
source_files
```

ملف `summary.html`:
- معرض محلي بسيط.
- صورة أفضل وجه.
- الوقت.
- الكاميرا.
- نوع التسجيل.
- رابط المقطع.
- رابط مجلد الظهور.

---

# 13. الأداء

المتطلبات:
- لا تحمل الفيديوهات كاملة في الذاكرة.
- عالج ملفاً بعد ملف.
- استخدم sample_fps.
- استخدم threading حتى لا تتجمد الواجهة.
- اجعل كل شيء قابلاً للاستكمال.
- لا تعيد معالجة ملف processed إلا إذا طلب المستخدم.
- استخدم GPU إذا توفر.
- CPU fallback لازم يعمل.

خيارات مهمة في الإعدادات:
```text
camera = front / rear / both
recording_types = N,E,I,M,P
sample_fps = 3
person_confidence = 0.35
face_confidence = 0.50
max_files = optional
date_from = optional
date_to = optional
```

---

# 14. الأخطاء التي يجب التعامل معها

تعامل مع:
```text
Missing SD card
No MP4 files found
Permission errors
Destination disk full
Corrupt MP4 file
FFmpeg not installed
OpenCV cannot decode H.265
Duplicate file names
Interrupted import
Interrupted processing
Malformed BlackVue filename
GPU unavailable
Model download failure
```

المبدأ:
- لا ينهار البرنامج بسبب ملف واحد.
- سجّل الخطأ.
- أكمل مع الملف التالي.
- اعرض ملخص الأخطاء في الواجهة.

---

# 15. الخصوصية

المتطلبات:
- كل شيء محلي.
- لا رفع للسحابة.
- لا API خارجي.
- لا face recognition بالاسم.
- لا مقارنة مع قواعد بيانات.
- لا مشاركة للنتائج.
- التحليل فقط لاستخراج أفضل صور الأشخاص الذين ظهروا في الفيديو.

أضف في README:
- البرنامج يتعامل مع صور أشخاص من فيديوهات السيارة.
- المستخدم مسؤول عن الاستخدام النظامي.
- يجب حفظ النتائج بأمان.
- يفضل حذف النتائج عند انتهاء الحاجة.

---

# 16. معايير قبول النسخة الأولى MVP

النسخة الأولى تعتبر ناجحة إذا:

1. يفتح البرنامج بواجهة رسومية.
2. يستطيع المستخدم اختيار SD card أو source folder.
3. يستطيع المستخدم اختيار archive folder.
4. يستطيع البرنامج فحص ملفات MP4 وعرض ملخص.
5. يستطيع البرنامج نسخ الملفات من SD إلى الجهاز دون تعديل SD.
6. يستطيع البرنامج استكمال النسخ إذا توقف.
7. يستطيع البرنامج إنشاء SQLite database.
8. يستطيع البرنامج تحليل أسماء ملفات BlackVue.
9. يستطيع البرنامج قراءة metadata باستخدام ffprobe.
10. يستطيع البرنامج معالجة 10 ملفات فيديو على الأقل.
11. يستطيع البرنامج اكتشاف أشخاص في الفيديو.
12. يستطيع البرنامج اكتشاف وجوه داخل قصاصات الأشخاص.
13. يستطيع البرنامج اختيار أفضل وجه حسب quality score.
14. يستطيع البرنامج حفظ best_face.jpg.
15. يستطيع البرنامج حفظ metadata.json.
16. يستطيع البرنامج إنشاء summary.csv.
17. يستطيع البرنامج إنشاء summary.html.
18. لا تتجمد الواجهة أثناء النسخ أو المعالجة.
19. إذا تلف ملف فيديو، يسجل الخطأ ويكمل.
20. يمكن فتح مجلد النتائج من الواجهة.

---

# 17. ترتيب التنفيذ المطلوب

نفذ بهذا الترتيب:

```text
1. Project setup
2. PySide6 main window
3. Import screen UI
4. BlackVue filename parser
5. SD scanner
6. SQLite database
7. Safe importer with progress
8. ffprobe metadata indexer
9. Processing settings UI
10. Frame sampler
11. Person detection
12. Face detection
13. Quality scoring
14. Save outputs
15. Results gallery
16. Clip exporter
17. HTML/CSV report
18. Error handling improvements
19. Packaging for Windows
```

---

# 18. ملاحظات مهمة للمطور

- الواجهة مهمة، لكن لا تضع منطق المعالجة داخل ملفات الواجهة.
- استخدم separation بين:
  - GUI
  - core processing
  - storage
  - utilities
- استخدم PySide6 Signals/Slots للتحديثات.
- استخدم QThread أو QRunnable للمهام الطويلة.
- لا تستخدم threading بطريقة تجعل OpenCV أو PyTorch يسببان تجمد الواجهة.
- استخدم logging إلى ملفات.
- استخدم pathlib لدعم مسارات Windows.
- اجعل كل خطوة قابلة للإعادة والاستكمال.
- لا تبدأ بتحسينات معقدة مثل clustering إلا بعد نجاح MVP.
- الأفضل أن تكون النسخة الأولى دقيقة بما يكفي وعملية، لا مثالية.

---

# 19. أمر تشغيل متوقع أثناء التطوير

```bash
python -m blackvue_person_extractor.app
```

أو:

```bash
python src/blackvue_person_extractor/app.py
```

---

# 20. المطلوب في README

اكتب README يحتوي:

```text
Project description
Installation steps
FFmpeg installation requirement
How to run the GUI
How to select SD card
How archive folders are created
Privacy notice
Limitations
Troubleshooting
```

حدود البرنامج:
```text
- قد لا يكتشف الوجوه البعيدة جداً.
- قد لا ينجح مع الوجوه الجانبية أو المغطاة.
- جودة الليل أقل.
- الانعكاسات والزجاج قد تقلل الدقة.
- البرنامج يستخرج أفضل لقطة ممكنة، لكنه لا يضمن استخراج كل شخص ظهر بالفيديو.
```

---

# 21. صياغة مختصرة تعطيها لـ Cursor في أول رسالة

```text
Build this as a GUI-first Windows Python app using PySide6. Do not make it CLI-first. The app should import BlackVue DR970X LTE Plus videos from an SD card to local storage, index the files, process them locally, detect people and faces, select the clearest face image for each appearance, export clips and reports, and keep the UI responsive. Follow the attached specification exactly. Start with a working GUI import flow and SQLite indexing before adding AI processing.
```
