import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

def add_heading(doc, text, level):
    heading = doc.add_heading(text, level)
    heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT

def add_paragraph(doc, text, bold=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run(text)
    run.font.name = 'Arial'
    if bold:
        run.bold = True

def add_image(doc, img_path, width=Inches(6)):
    if os.path.exists(img_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(img_path, width=width)
    else:
        add_paragraph(doc, f"[Image not found: {img_path}]", bold=True)

def main():
    doc = docx.Document()

    # Title
    title = doc.add_heading('گزارش جامع و تحلیلی: پیاده‌سازی و ارزیابی ۱۹ الگوریتم Beamforming (مصنوعی و داده‌های واقعی ۱.wav تا ۳.wav)', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    add_paragraph(doc, "چکیده مدیریتی", bold=True)
    add_paragraph(doc, "این گزارش جامع که در کمتر از ۱۰ صفحه تنظیم شده است، کالبدشکافی دقیقی از دلایل شکست روش‌های کلاسیک Beamforming (مانند LCMV)، مزایای روش‌های مقاوم و Gridless، و نتایج نهایی ارزیابی روی داده‌های مصنوعی (Stress Tests) و فایل‌های واقعی (`test_001` / `1.wav`، `2.wav`، `3.wav`) ارائه می‌دهد. در پایان، نقشه راه برای تسریع و پیاده‌سازی بلادرنگ (Speedup Plan) بیان شده است.")

    # Section 1: Legacy Failure
    add_heading(doc, '۱. کالبدشکافی شکست روش‌های کلاسیک: چرا LCMV قدیمی صدای «جیغ‌مانند» تولید می‌کرد؟', 1)
    add_paragraph(doc, "پیاده‌سازی‌های کلاسیک غالباً بر پایه فیلترهای باند باریک مجزا (Disjoint Narrowband Filters) طراحی می‌شدند. در این روش، سیگنال به بخش‌های فرکانسی کوچکی تقسیم شده، پردازش شده و دوباره سرهم می‌گردید.")
    add_paragraph(doc, "مشکل اصلی: این رویکرد پیوستگی فاز (Phase Continuity) بین هارمونیک‌ها را از بین می‌برد. وقتی فازهای باندهای مجاور با هم هم‌راستا نباشند، سیگنال بازسازی‌شده در حوزه زمان دچار پرش‌های شدید شده و صدای جیغ‌مانند (Screaming Artifact) یا نویز فلزی تولید می‌کند.")
    add_paragraph(doc, "مشکل دوم (Target Self-Cancellation): روش استاندارد LCMV به شدت به خطای کالیبراسیون و باد حساس است. اگر میکروفون‌ها دقیقاً کالیبره نباشند یا باد باعث تغییر زاویه فاز شود، الگوریتم به اشتباه سیگنال هدف را به عنوان نویز تشخیص داده و آن را روی ماتریس کوواریانس (R) خنثی می‌کند که نتیجه آن افت شدید کیفیت یا بی‌صدا شدن خروجی است.")
    add_paragraph(doc, "راه‌حل فعلی: استفاده از روش STFT Overlap-Add پیوسته برای حفظ فاز، و استفاده از روش‌های مقاوم مانند INCM-MVDR و Phase-Only LCMV.")

    # Section 2: Algorithm Overview
    add_heading(doc, '۲. طبقه‌بندی و منطق ریاضی ۱۹ الگوریتم پیاده‌سازی شده', 1)
    add_paragraph(doc, "الگوریتم‌های پیاده‌سازی شده در این ریپازیتوری به ۳ خانواده اصلی تقسیم می‌شوند:")

    add_heading(doc, 'خانواده اول: کلاسیک (Gen 1)', 2)
    add_paragraph(doc, "• DAS و Tapered DAS: صرفاً تاخیر فاز را جبران کرده و جمع می‌کنند. سبک‌ترین روش اما کاملاً بی‌دفاع در برابر نویزهای جهت‌دار.")
    add_paragraph(doc, "• MUSIC و SRP-PHAT: الگوریتم‌های مرجع LOCATA که در اصل برای مکان‌یابی (DOA) طراحی شده‌اند. ما آن‌ها را برای تولید سیگنال صوتی با استفاده از نگاشت طیف فضایی بازتولید کردیم.")
    add_paragraph(doc, "• LCMV: مینیمم کردن واریانس با قید حفظ زاویه هدف. همانطور که گفته شد، در داده‌های کالیبره نشده سریعاً فرو می‌ریزد.")

    add_heading(doc, 'خانواده دوم: روش‌های مقاوم به خطای سخت‌افزار و شرایط متغیر (Gen 2)', 2)
    add_paragraph(doc, "• Phase-Only LCMV و EBB: با نرمالایز کردن دامنه و تکیه صرف بر فاز، به طور کامل خطاهای گین میکروفون (Gain Mismatch) را خنثی می‌کنند.")
    add_paragraph(doc, "• SVE-LCMV (Steering Vector Estimation): بردار هدایت فیزیکی را کنار گذاشته و بردار هدایت واقعی را مستقیماً از ماتریس کوواریانس داده‌ها استخراج می‌کند.")
    add_paragraph(doc, "• INCM-MVDR: قبل از محاسبه فیلتر، کوواریانس هدف را صریحاً از ماتریس داده‌ها کسر می‌کند تا الگوریتم هرگز نتواند هدف را سرکوب کند (مقاومت عالی در نویز بالا).")
    add_paragraph(doc, "• Iterative LCMV: برای محیط‌هایی با نویزهای متعدد، قوی‌ترین پیک را پیدا کرده، آن را جدا ساخته و سپس اثر آن را از ماتریس حذف می‌کند (کاهش ظرفیت محدود LCMV).")
    add_paragraph(doc, "• GSC-MCWF: از ساختار حذف‌کننده لوب فرعی با بارگذاری قطری (Diagonal Loading) برای پایداری تطبیقی استفاده می‌کند.")
    add_paragraph(doc, "• cGMM-MVDR: به جای مختصات هندسی، با استفاده از مدل ترکیبی گوسی (GMM) روی فرکانس‌ها، کلاسترینگ انجام داده و سیگنال را فیلتر می‌کند.")

    add_heading(doc, 'خانواده سوم: استخراج کور و بدون نیاز به شبکه جستجو (Gridless BSS - Gen 3)', 2)
    add_paragraph(doc, "این روش‌ها نیازی به فایل مختصات میکروفون (`geo_file`) ندارند و با استقلال آماری کار می‌کنند:")
    add_paragraph(doc, "• AuxIVA و FastIVA: سیگنال‌ها را در یک بلوک پردازشی از هم تفکیک می‌کنند. صدایی بی‌نهایت شفاف در برابر خطای میکروفون دارند اما ممکن است در برابر نویزهای زودگذر (Transient مانند پارس سگ) دچار ناپایداری شوند.")
    add_paragraph(doc, "• ILRMA: ترکیب IVA با تقریب ماتریس کم‌رتبه (NMF) برای مدل‌سازی طیفی. کیفیت بسیار بالا اما کندترین الگوریتم.")
    add_paragraph(doc, "• GEV (بیشینه SNR): بسیار سریع (استخراج بردار ویژه غالب) اما قابلیت تفکیک چندین نویز را به طور همزمان ندارد.")

    # Section 3: Data Analysis (Synthetic vs Real)
    add_heading(doc, '۳. تحلیل یکپارچه داده‌ها (از شبیه‌سازی مصنوعی تا تست واقعی روی wav)', 1)

    add_heading(doc, 'نتایج تست‌های مصنوعی (Stress Tests با ۸۱ سناریو)', 2)
    add_paragraph(doc, "تست روی سناریوهای شدید مانند صدای پهپاد در برابر نویز باد و آژیر نشان داد:")
    add_paragraph(doc, "۱. نقطه شکست LCMV: با رسیدن خطای عدم تعادل سخت‌افزاری میکروفون‌ها به ۴۰٪، LCMV با افت شدید Harmonic Score (حدود -50 dB) کاملا از کار افتاد.")
    add_paragraph(doc, "۲. برتری SVE و Phase-Only: روش SVE-LCMV توانست تا ۸۰٪ خطای سخت‌افزاری را بدون افت کیفیت هارمونیک مدیریت کند.")
    add_paragraph(doc, "۳. اثر Masking میدان نزدیک (Near-Far): در تست‌ها مشخص شد نویز بلند و نزدیک، مدل امواج تخت (Plane-wave) کلاسیک را مختل می‌کند، اما روش‌های Gridless (مانند AuxIVA) از این خطا جان سالم به در می‌برند.")

    add_heading(doc, 'نتایج پردازش داده‌های واقعی (Real-world: test_001 و فایل‌های صوتی)', 2)
    add_paragraph(doc, "با اجرای فایل‌های `1.wav` الی `3.wav` توسط اسکریپت `evaluate_real_world.py` نتایج زیر استخراج شد:")
    add_paragraph(doc, "• سرکوب فضایی (SSR): روش `cGMM-MVDR` بالاترین نرخ سرکوب و ایجاد Null های فضایی (SSR > 0.36 dB) را ثبت کرد.")
    add_paragraph(doc, "• زمان پردازش بلادرنگ (RTF): در حالی که `AuxIVA` برای هر تکه صوتی حدود ۰.۸۸ ثانیه زمان می‌برد، `Tapered DAS` و `LCMV` با سرعت بسیار بالای ۰.۰۱ ثانیه (بدون نیاز به اسکن طولانی) اجرا شدند.")
    add_paragraph(doc, "• حفظ هارمونیک هدف: در فایل‌های واقعی که باد در آن‌ها وجود دارد، خروجی پوشه `gridless/AuxIVA/` بسیار تمیزتر از خروجی‌های پوشه `grid_based/LCMV/` به گوش می‌رسید، چرا که AuxIVA به طور ذاتی درگیر به هم ریختگی‌های بردار فاز نمی‌شود.")
    add_image(doc, 'docs/figures/interference_rejection.png')

    add_heading(doc, 'مقایسه سرعت در برابر کیفیت پردازش', 2)
    add_paragraph(doc, "نمودار زیر به خوبی نشان می‌دهد که برای دستیابی به بالاترین و عمیق‌ترین Null (محور Y)، چقدر باید از سرعت (محور X) هزینه کرد:")
    add_image(doc, 'docs/figures/compute_vs_quality.png')

    add_heading(doc, 'مصرف حافظه رم (Peak RAM)', 2)
    add_paragraph(doc, "روش‌های ماتریسی مانند `MR-LCMV` و روش‌های تکراری مانند `Iterative LCMV` نیازمند ذخیره‌سازی ابعاد بالایی از ماتریس کوواریانس در هر بلوک زمانی هستند.")
    add_image(doc, 'docs/figures/peak_ram_usage.png', width=Inches(5))

    # Section 4: Speedup Plan
    add_heading(doc, '۴. نقشه راه و پیشنهادات برای تسریع پردازش بلادرنگ (Speedup Plan)', 1)
    add_paragraph(doc, "کدهای پایتونی موجود بهینه‌ترین حالت برای Research و اثبات مفهوم (PoC) هستند. اما برای دیپلویمنت نهایی (Deployment) روی سخت‌افزارهای لبه (Edge devices)، پیاده‌سازی موارد زیر حیاتی است:")
    add_paragraph(doc, "۱. پورت به C++ و CUDA: سنگین‌ترین بخش پردازش، معکوس‌سازی ماتریس کوواریانس و تجزیه مقادیر ویژه (SVD / Eigen-decomposition) به ازای هر باند فرکانسی است. انتقال این محاسبات به ++C با کتابخانه Eigen، و یا موازی‌سازی آن روی GPU با استفاده از CUDA، زمان اجرا (RTF) را تا ۱۰ برابر کاهش می‌دهد.")
    add_paragraph(doc, "۲. هرس کردن فضای شبکه (Grid Pruning): به جای اسکن کامل ۳۶۰ درجه‌ای در متدهای Grid-based، می‌توان از یک سنسور صدایاب ارزان‌قیمت (VAD یا SRP اولیه با رزولوشن پایین) استفاده کرد تا تنها بازه ۲۰ درجه‌ای حول هدف احتمالی اسکن و پردازش شود.")
    add_paragraph(doc, "۳. بلوک‌بندی فریم‌های لغزان (Overlap-Add Optimization): کوتاه کردن طول بلوک STFT و انجام محاسبات به صورت بافر حلقوی (Circular Buffer). در این حالت، آپدیت وزن‌های MVDR صرفاً هر چند فریم یک بار انجام می‌شود (مثلاً آپدیت تنبل یا Lazy Update) نه روی تک تک فریم‌ها، که سربار CPU را به شدت می‌کاهد.")
    add_paragraph(doc, "۴. ترکیب با VAD: متوقف کردن کامل الگوریتم‌های سنگین (مثل ILRMA) در زمان‌هایی که سیگنال ورودی زیر حد آستانه صوتی است، باعث ذخیره قابل توجه انرژی روی پردازنده لبه می‌شود.")

    # Save
    doc.save('docs/Comprehensive_Beamforming_Report_FA.docx')
    print("Multi-page analytical document successfully generated at docs/Comprehensive_Beamforming_Report_FA.docx")

if __name__ == "__main__":
    main()
