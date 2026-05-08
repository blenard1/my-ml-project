from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt


EMU_PER_INCH = 914400
SLIDE_W = 13.333
SLIDE_H = 7.5
CX = int(SLIDE_W * EMU_PER_INCH)
CY = int(SLIDE_H * EMU_PER_INCH)


@dataclass
class TextBox:
    text: str
    x: float
    y: float
    w: float
    h: float
    size: int = 24
    bold: bool = False
    color: str = "1F2937"


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    fill: str
    line: str = "FFFFFF"


@dataclass
class SlideSpec:
    title: str
    subtitle: str
    bullets: list[str]
    accent: str
    footer: str


SLIDES = [
    SlideSpec(
        title="Salient Object Detection",
        subtitle="End-to-end CNN pipeline from scratch",
        bullets=[
            "Input RGB image",
            "One-channel saliency mask",
            "Overlay for visual inspection",
        ],
        accent="2563EB",
        footer="Project #3 | ML/DL",
    ),
    SlideSpec(
        title="Dataset and Pipeline",
        subtitle="ECSSD image/mask pairs with deterministic train/val/test splits",
        bullets=[
            "Paired loader matches images and masks by filename stem",
            "Resize to 128 x 128 or 224 x 224",
            "Horizontal flip, random crop, brightness, and contrast jitter",
            "Masks stay aligned with nearest-neighbor resizing",
        ],
        accent="059669",
        footer="data_loader.py",
    ),
    SlideSpec(
        title="CNN Architecture",
        subtitle="From-scratch encoder-decoder with skip connections",
        bullets=[
            "Four Conv2D encoder levels with ReLU, MaxPool, and BatchNorm",
            "U-Net-style skip connections preserve object boundaries",
            "ConvTranspose decoder upsamples back to 128 x 128",
            "Sigmoid output produces a one-channel saliency map",
        ],
        accent="DC2626",
        footer="sod_model.py",
    ),
    SlideSpec(
        title="Results",
        subtitle="Improved model beats the baseline and supports high precision",
        bullets=[
            "Balanced mode: IoU 0.5364, F1 0.6915, recall 0.7711",
            "High-precision mode: precision 0.8726 at threshold 0.94",
            "Baseline F1 improved from 0.6278 to 0.6915",
            "Logs, checkpoints, metrics, and visual grids are saved",
        ],
        accent="7C3AED",
        footer="checkpoints/improved + artifacts/evaluation",
    ),
    SlideSpec(
        title="Demo and Next Steps",
        subtitle="Upload image -> mask, overlay, and inference time",
        bullets=[
            "Streamlit app loads the improved checkpoint by default",
            "Notebook shows the same inference workflow",
            "Report PDF and five-slide deck are included",
            "Repository is ready to share with the mentor",
        ],
        accent="D97706",
        footer="app.py + demo_notebook.ipynb",
    ),
]


def emu(value: float) -> int:
    return int(value * EMU_PER_INCH)


def paragraph(text: str, size: int, color: str, bold: bool = False) -> str:
    bold_xml = "<a:b/>" if bold else ""
    return (
        "<a:p><a:r><a:rPr lang=\"en-US\" sz=\""
        f"{size * 100}\" dirty=\"0\"><a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill>{bold_xml}</a:rPr>"
        f"<a:t>{escape(text)}</a:t></a:r></a:p>"
    )


def text_shape(shape_id: int, box: TextBox) -> str:
    return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(box.x)}" y="{emu(box.y)}"/><a:ext cx="{emu(box.w)}" cy="{emu(box.h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          <a:noFill/><a:ln><a:noFill/></a:ln>
        </p:spPr>
        <p:txBody><a:bodyPr wrap="square" rtlCol="0"/><a:lstStyle/>{paragraph(box.text, box.size, box.color, box.bold)}</p:txBody>
      </p:sp>
    """


def rect_shape(shape_id: int, rect: Rect) -> str:
    return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="Rectangle {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(rect.x)}" y="{emu(rect.y)}"/><a:ext cx="{emu(rect.w)}" cy="{emu(rect.h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          <a:solidFill><a:srgbClr val="{rect.fill}"/></a:solidFill>
          <a:ln><a:solidFill><a:srgbClr val="{rect.line}"/></a:solidFill></a:ln>
        </p:spPr>
      </p:sp>
    """


def slide_xml(spec: SlideSpec, number: int) -> str:
    shapes: list[str] = []
    next_id = 2
    shapes.append(rect_shape(next_id, Rect(0, 0, 13.333, 7.5, "F8FAFC", "F8FAFC")))
    next_id += 1
    shapes.append(rect_shape(next_id, Rect(0, 0, 0.28, 7.5, spec.accent, spec.accent)))
    next_id += 1
    shapes.append(text_shape(next_id, TextBox(spec.title, 0.9, 0.82, 7.8, 0.78, size=36, bold=True, color="111827")))
    next_id += 1
    shapes.append(text_shape(next_id, TextBox(spec.subtitle, 0.93, 1.68, 7.9, 0.48, size=17, color="475569")))
    next_id += 1

    y = 2.6
    for bullet in spec.bullets:
        shapes.append(rect_shape(next_id, Rect(1.02, y + 0.12, 0.12, 0.12, spec.accent, spec.accent)))
        next_id += 1
        shapes.append(text_shape(next_id, TextBox(bullet, 1.35, y, 7.8, 0.38, size=18, color="1F2937")))
        next_id += 1
        y += 0.64

    shapes.append(text_shape(next_id, TextBox(spec.footer, 0.93, 6.83, 5.0, 0.32, size=11, color="64748B")))
    next_id += 1
    shapes.append(text_shape(next_id, TextBox(str(number), 12.15, 6.82, 0.3, 0.3, size=12, bold=True, color="64748B")))

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{CX}" cy="{CY}"/><a:chOff x="0" y="0"/><a:chExt cx="{CX}" cy="{CY}"/></a:xfrm></p:grpSpPr>
      {''.join(shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def content_types() -> str:
    overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(SLIDES) + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {overrides}
</Types>
"""


def presentation_xml() -> str:
    slide_ids = "\n".join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, len(SLIDES) + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(SLIDES) + 1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="{CX}" cy="{CY}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""


def presentation_rels() -> str:
    rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, len(SLIDES) + 1)
    )
    rels += f'\n<Relationship Id="rId{len(SLIDES) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    rels += f'\n<Relationship Id="rId{len(SLIDES) + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {rels}
</Relationships>
"""


def root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def app_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{len(SLIDES)}</Slides>
</Properties>
"""


def core_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Salient Object Detection Project</dc:title><dc:creator>Blenard Tahiraj</dc:creator>
</cp:coreProperties>
"""


def theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="SOD Theme">
  <a:themeElements>
    <a:clrScheme name="SOD"><a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="059669"/></a:accent2><a:accent3><a:srgbClr val="DC2626"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="D97706"/></a:accent5><a:accent6><a:srgbClr val="64748B"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Default"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>
"""


def slide_master_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{CX}" cy="{CY}"/><a:chOff x="0" y="0"/><a:chExt cx="{CX}" cy="{CY}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>
"""


def slide_layout_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{CX}" cy="{CY}"/><a:chOff x="0" y="0"/><a:chExt cx="{CX}" cy="{CY}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""


def empty_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""


def slide_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>
"""


def write_pptx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as pptx:
        pptx.writestr("[Content_Types].xml", content_types())
        pptx.writestr("_rels/.rels", root_rels())
        pptx.writestr("docProps/app.xml", app_xml())
        pptx.writestr("docProps/core.xml", core_xml())
        pptx.writestr("ppt/presentation.xml", presentation_xml())
        pptx.writestr("ppt/_rels/presentation.xml.rels", presentation_rels())
        pptx.writestr("ppt/theme/theme1.xml", theme_xml())
        pptx.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        pptx.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>')
        pptx.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        pptx.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", empty_rels())
        for i, spec in enumerate(SLIDES, start=1):
            pptx.writestr(f"ppt/slides/slide{i}.xml", slide_xml(spec, i))
            pptx.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels())


def write_previews(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(SLIDES, start=1):
        fig, ax = plt.subplots(figsize=(13.333, 7.5))
        ax.set_facecolor("#F8FAFC")
        ax.add_patch(plt.Rectangle((0, 0), 0.28, 7.5, color=f"#{spec.accent}"))
        ax.text(0.9, 6.52, spec.title, fontsize=36, fontweight="bold", color="#111827", va="top")
        ax.text(0.93, 5.82, spec.subtitle, fontsize=17, color="#475569", va="top")
        y = 5.0
        for bullet in spec.bullets:
            ax.add_patch(plt.Rectangle((1.02, y - 0.06), 0.12, 0.12, color=f"#{spec.accent}"))
            ax.text(1.35, y, bullet, fontsize=18, color="#1F2937", va="center")
            y -= 0.64
        ax.text(0.93, 0.38, spec.footer, fontsize=11, color="#64748B", va="bottom")
        ax.text(12.15, 0.42, str(index), fontsize=12, fontweight="bold", color="#64748B", va="bottom")
        ax.set_xlim(0, 13.333)
        ax.set_ylim(0, 7.5)
        ax.axis("off")
        fig.savefig(output_dir / f"slide_{index}.png", dpi=140, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def write_pdf(preview_dir: Path, pdf_path: Path) -> None:
    images = []
    for index in range(1, len(SLIDES) + 1):
        image_path = preview_dir / f"slide_{index}.png"
        images.append(plt.imread(image_path))

    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for image in images:
            fig, ax = plt.subplots(figsize=(13.333, 7.5))
            ax.imshow(image)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pptx_path = root / "artifacts" / "presentation_slides.pptx"
    preview_dir = root / "artifacts" / "previews"
    pdf_path = root / "artifacts" / "presentation_slides.pdf"
    write_pptx(pptx_path)
    write_previews(preview_dir)
    write_pdf(preview_dir, pdf_path)
    print(f"Wrote {pptx_path}")
    print(f"Wrote previews to {preview_dir}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
