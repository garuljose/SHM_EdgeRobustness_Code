const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ExternalHyperlink,
  AlignmentType, BorderStyle, UnderlineType, SectionType, Table, TableRow,
  TableCell, WidthType, ImageRun, ShadingType, VerticalAlign,
  Math: Mth, MathRun, MathFraction, MathSuperScript, MathSubScript, MathRadical, MathRoundBrackets,
} = require("docx");

const FONT = "Times New Roman";
const OUT = "outputs";

// ---------------------------------------------------------------- helpers
function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: 20, ...opts }); // 20 half-points = 10pt
}
function para(children, opts = {}) {
  return new Paragraph({ children, spacing: { after: 90 }, alignment: AlignmentType.JUSTIFIED, ...opts });
}
function body(text, opts = {}) {
  return para([run(text)], opts);
}
function sectionHeading(numeral, title) {
  return new Paragraph({
    children: [run(`${numeral}. ${title}`, { bold: true, size: 21 })],
    spacing: { before: 200, after: 100 },
    alignment: AlignmentType.CENTER,
  });
}
function subHeading(text) {
  return new Paragraph({
    children: [run(text, { bold: true, italics: true, size: 20 })],
    spacing: { before: 100, after: 60 },
  });
}
function figureImage(path, widthPx, caption, label) {
  const data = fs.readFileSync(path);
  return [
    new Paragraph({
      children: [new ImageRun({ data, transformation: { width: widthPx.w, height: widthPx.h }, type: "png" })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 40 },
    }),
    new Paragraph({
      children: [run(`${label}. ${caption}`, { size: 17, italics: false })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 160 },
    }),
  ];
}
function cite(n) {
  return `[${n}]`;
}

const NOBORDER = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
// Native Word (OOXML/OMML) display equation, right-numbered, IEEE style.
function eqBlock(mathChildren, label) {
  return new Table({
    width: { size: 4860, type: WidthType.DXA },
    columnWidths: [4300, 560],
    borders: { top: NOBORDER, bottom: NOBORDER, left: NOBORDER, right: NOBORDER, insideHorizontal: NOBORDER, insideVertical: NOBORDER },
    rows: [new TableRow({ children: [
      new TableCell({
        width: { size: 4300, type: WidthType.DXA }, margins: { top: 40, bottom: 40, left: 0, right: 0 },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new Mth({ children: mathChildren })] })],
      }),
      new TableCell({
        width: { size: 560, type: WidthType.DXA }, margins: { top: 40, bottom: 40, left: 0, right: 0 }, verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [run(label)] })],
      }),
    ] })],
  });
}
function mfrac(numChildren, denChildren) { return new MathFraction({ numerator: numChildren, denominator: denChildren }); }
function msup(base, sup) { return new MathSuperScript({ children: [new MathRun(base)], superScript: [new MathRun(sup)] }); }
function msub(base, sub) { return new MathSubScript({ children: [new MathRun(base)], subScript: [new MathRun(sub)] }); }
function mr(t) { return new MathRun(t); }

function makeTable(headerCells, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  const cellMargins = { top: 20, bottom: 20, left: 40, right: 40 };
  const headerRow = new TableRow({
    children: headerCells.map((h, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      margins: cellMargins,
      shading: { type: ShadingType.CLEAR, color: "auto", fill: "D9D9D9" },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ children: [run(h, { bold: true, size: 15 })], alignment: AlignmentType.CENTER })],
    })),
  });
  const bodyRows = rows.map(r => new TableRow({
    children: r.map((c, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      margins: cellMargins,
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ children: [run(String(c), { size: 15 })], alignment: AlignmentType.CENTER })],
    })),
  }));
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...bodyRows],
  });
}
function tableCaption(label, text) {
  return new Paragraph({
    children: [run(`${label}`, { bold: false, size: 16 }), run(`\n${text}`, { size: 16 })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 140, after: 60 },
  });
}

// ================================================================== CONTENT
const title = "Resource-Efficient Self-Supervised Classification for Edge-Based Structural Health Monitoring Under Corrupted Sensor Data";
const authors = "G. Jai Arul Jose";
const affil = "Department of Computer Science and Engineering, BMS Institute of Technology & Management (BMSIT&M), Bengaluru, Karnataka, India";

const abstractText = "Deep learning has, within a short span, become the default tool for vibration-based structural health monitoring (SHM); yet the majority of reported models are validated under an assumption that few field engineers would accept — that sensor data arrives clean and complete, and that inference may draw on effectively unconstrained computing resources. Real deployments violate both assumptions. Field sensors drop packets, saturate during precisely the high-amplitude events that carry the most damage-relevant information, and battery-powered nodes brown out; at the same time, continuous monitoring at scale is pushing inference away from the cloud and onto the sensor node itself, where memory and latency are tightly rationed. This paper studies the resulting trade-off directly, asking not how faithfully a corrupted signal can be reconstructed but how much task accuracy a resource-constrained model retains once missingness and noise are actually present. Owing to network restrictions in the computing environment used for this study, standard recorded SHM archives could not be reached; a numerically simulated four-degree-of-freedom shear-building benchmark is used in their place, a limitation stated candidly in Section IX rather than concealed. Four classifiers — a full-capacity spectral network, a linear-interpolation baseline, a GRU-based reconstruct-then-classify baseline representative of recent imputation literature, and a proposed compact classifier pretrained through self-supervised masked-spectrum reconstruction — are compared across three canonical missingness mechanisms (MCAR, MAR, MNAR) at four corruption rates. The proposed model requires 858 parameters against 36,772 for the reconstruction baseline, a 42.9-fold reduction, and completes inference in 0.21 ms per window against 27.6 ms, a 131-fold speed-up, at an average accuracy cost of 4.2 percentage points. Measured on an accuracy-per-kilobyte basis, however, it is roughly twice as efficient as the full-capacity baseline and six times more efficient than the reconstruction baseline, and under the field-realistic burst-dropout (MAR) mechanism its accuracy degrades far more gracefully, overtaking the full-capacity baseline beyond 30% missingness. Self-supervised pretraining alone contributes a 3.3-point average accuracy gain over training the same compact architecture from scratch, while post-training dynamic quantisation is shown to help the GRU-heavy baseline substantially and the already-compact proposed model not at all — a modest but honestly reported negative result. All code is released for reproduction on a single laptop or a free-tier cloud notebook.";

const keywords = "structural health monitoring; edge AI; TinyML; missing data; sensor noise; self-supervised learning; model quantisation; vibration-based damage detection; MCAR; MAR; MNAR";

// ---------------- Intro
const introParas = [
  "Vibration-based structural health monitoring (SHM) infers the condition of a bridge, building, or other civil structure from accelerometer or strain time-series, relying on shifts in modal or spectral features as the damage-sensitive signal. Over roughly the last three years the field has moved decisively toward deep learning, and recent comprehensive reviews confirm that the central question has shifted from whether such models work to how they ought to be deployed " + cite(1) + ", " + cite(2) + ", " + cite(3) + ". Two deployment realities are, in the authors' view, becoming difficult to ignore. First, field sensor networks seldom deliver clean, complete signals: wireless links drop packets, batteries brown out, and channels saturate during precisely the high-amplitude events that carry the most damage-relevant information. Second, continuous monitoring at scale is pushing inference away from the cloud and onto the sensor node itself, where memory, power, and latency are all tightly bounded " + cite(4) + ", " + cite(5) + ", " + cite(6) + ", " + cite(7) + ".",
  "Recent imputation research has responded seriously to the first of these problems, reconstructing corrupted sensor channels with increasingly sophisticated sequence models " + cite(1) + ". What such work optimises for, however, is fidelity to the original waveform, measured under effectively unconstrained compute — a reasonable objective offline, but arguably the wrong one once the endpoint is a real-time decision made on constrained hardware. A model deployed at the edge does not need to reconstruct a waveform for a human to inspect; it needs to arrive at a correct classification from whatever signal survives. Self-supervised representation learning has, separately, begun to be explored for SHM in its own right, with label-free pretraining shown to disentangle genuine damage signatures from operational and environmental variability without reconstruction as an explicit downstream objective " + cite(8) + ". The present study sits at the intersection of these two threads and asks the narrower, and arguably more consequential, question: how does damage-classification accuracy degrade as a function of missingness type and rate, for models actually sized to fit on edge hardware, and does a resource-constrained model trained directly for the task close the gap to a full-capacity reconstruction pipeline once both are evaluated under matched, real-time constraints?",
  "To answer this question, a four-degree-of-freedom (4-DOF) shear-building structure is simulated numerically, a standard, low-order representation of the class of benchmark structures long used in the SHM machine-learning literature, such as the LANL three-storey structure and the IASC–ASCE benchmark families. Three canonical missing-data mechanisms are injected at controlled rates, and four classifiers spanning a wide range of size and reconstruction sophistication are compared under matched conditions. The outcome is a quantified accuracy-versus-resource trade-off, expressed both in raw percentage points and as an explicit efficiency ratio (Section VII), an ablation isolating the contribution of self-supervised pretraining, and a candid account of when post-training quantisation does, and does not, help.",
];

// ---------------- Lit review
const litParas = [
  "Established consensus. Deep learning for SHM is, at this point, mainstream rather than experimental. Convolutional and recurrent architectures reliably outperform manual inspection for vibration-based damage detection under controlled conditions, and this finding is echoed across structural domains as different as aerospace and, more recently, digital-twin-integrated civil condition monitoring " + cite(2) + ", " + cite(3) + ", suggesting that the underlying data-scarcity and generalisation challenges are properties of the problem itself rather than of any single benchmark structure.",
  "Missing and noisy data. Sensor dropout and corruption are now widely acknowledged as a central, unresolved bottleneck rather than a rare edge case. Tran The et al. " + cite(1) + " reconstruct missing SHM channels using a gated dilated convolution and GRU architecture, reporting strong reconstruction fidelity; this is the most directly comparable prior work to the present study and is reimplemented here as the Recon-GRU baseline. A largely separate, statistics-oriented literature has continued to refine the MCAR/MAR/MNAR taxonomy itself, with recent comparative work confirming that imputation accuracy varies sharply by mechanism and by the non-stationarity of the underlying series " + cite(9) + " — precisely the sensitivity this paper exploits by evaluating all three mechanisms rather than one. What the reconstruction-focused strand of this literature does not examine is whether the resulting fidelity gains survive translation into a downstream decision made under a strict resource budget.",
  "Edge and resource-constrained deployment. A 2025 review of edge computing specifically within SHM documents the shift towards on-device inference and the drivers behind it " + cite(4) + ", and TinyML-based real-time anomaly detection has been demonstrated for civil infrastructure on microcontroller-class hardware " + cite(5) + ", with a closely related 2026 study demonstrating an efficient TinyML pipeline for UAV-based crack classification under a similarly tight resource budget " + cite(7) + ". A 2025 secure edge-computing reference architecture for SHM is candid about the resulting engineering trade-offs but treats them primarily as a systems and security concern " + cite(6) + " rather than a modelling one; separately, a 2025 comparative study of pruning and quantisation across standard and custom neural networks confirms that the two compression strategies are not interchangeable and that the better choice depends materially on the target architecture " + cite(10) + " — a finding this paper independently arrives at for quantisation alone (Section VII). None of this body of work characterises model accuracy jointly against a resource budget under corrupted input, which is the gap the present study targets directly.",
  "Generalisation and physics-informed alternatives. Domain shift between training and deployment settings remains well documented outside civil SHM as well: a 2025 review of machine learning for aerospace SHM reports the identical generalisation failure mode in an entirely different structural domain " + cite(2) + ", and meta-learning has been proposed as a partial remedy for the associated scarcity of labelled real-world data " + cite(11) + ". Physics-informed neural networks offer a parallel route towards more data-efficient, generalisable models by embedding governing equations directly into training " + cite(12) + "; the present study does not adopt a physics-informed loss, since its classification target is empirical rather than a governed physical quantity, but combining the two ideas — physics-informed structure with the self-supervised objective used here — is a natural direction for future work.",
  "Reproducibility. Machine learning research more broadly faces a documented reproducibility gap, driven by undisclosed hyperparameters, non-public code, and narrow benchmark reuse " + cite(13) + "; SHM's own heavy reliance on a small number of shared benchmark structures and datasets plausibly compounds this difficulty, a connection this paper makes explicit rather than merely assumes, and addresses directly by releasing all code and by reporting the single-run, no-hardware limitations of the present study candidly (Section IX).",
];

// ---------------- Contribution
const contribBullets = [
  "A reframing of the missing/noisy-sensor-data problem in SHM around task-relevant accuracy per unit of resource, rather than signal-reconstruction fidelity under unconstrained compute, formalised through an explicit accuracy-per-kilobyte efficiency measure given in (7).",
  "A controlled comparison of four classifiers, including a reimplementation of a recent reconstruction-based baseline " + cite(1) + ", across three standard missingness mechanisms (MCAR, MAR, MNAR) and four corruption rates, all evaluated under matched inference conditions.",
  "An ablation isolating the contribution of self-supervised masked-spectrum pretraining, and a candid, non-inflated account of when post-training dynamic quantisation does and does not reduce model footprint.",
  "A fully reproducible, laptop- or free-tier-Colab-executable open-source pipeline (simulation, corruption injection, training, evaluation, and figure generation), released in substitution for the recorded field datasets that could not be reached from this study's execution environment.",
];

// ---------------- Experimental setup (paragraph split around Eq. 5/6)
const setupPara1 = "All experiments are run on a 2-core CPU with no GPU, in PyTorch 2.x, so as to reflect the laptop or free-tier-notebook constraint this study targets. Models are trained with Adam: the full-signal baseline for 35 epochs (lr = 2×10⁻³), the reconstruction baseline for 25 epochs (lr = 1.5×10⁻³) with mean-squared-error loss against the clean signal, the self-supervised encoder for 25 epochs (lr = 2×10⁻³), and the classification fine-tuning stage for 35 epochs (lr = 3×10⁻³) with cross-entropy loss; the best validation-accuracy checkpoint is retained in each case. Damage classification (healthy versus damaged) is reported in terms of accuracy and F1, computed as in (5) and (6),";
const setupPara2 = "together with the area under the receiver-operating-characteristic curve (AUC), alongside model size (KB), parameter count, and per-window CPU inference latency (ms) as the paired resource-level metrics. All reported results reflect a single simulation seed and a single training run per model, a limitation made explicit in Section IX.";

// ---------------- Results
const resultsParas = [
  "Table II reports accuracy, F1, and AUC for all five models, averaged across the 12 corrupted test conditions (3 mechanisms × 4 rates) and separately on clean data. It may be observed that the Naive-impute baseline is, on raw accuracy, the strongest performer overall (82.7%), narrowly ahead of Recon-GRU (81.1%) and the Full-signal baseline (80.0%); this is itself a useful, honestly reported finding, since a simple, near-zero-cost interpolation step is seen to capture most of the benefit that a considerably heavier reconstruction network provides in this setting. The Proposed model trails on this measure, at 76.9% average accuracy, a 4.2-point gap to Recon-GRU. Fig. 1 shows, however, that this headline figure conceals an important pattern: under MAR (burst dropout), the Full-signal baseline degrades from 83.6% at 10% missingness to 75.6% at 40%, whereas the Proposed model remains essentially flat (77.8% to 78.1%) and overtakes the Full-signal baseline beyond the 30% missingness rate. A similar, if smaller, graceful-degradation pattern is seen under MNAR. Under MCAR, by contrast, all baselines remain comfortably ahead of the Proposed model across the full rate range, which is consistent with MCAR being the least field-realistic of the three corruption mechanisms considered.",
  "Fig. 4 confirms that the simulated damage manipulation is physically grounded and consistent with (4): the first three elastic modes shift downward monotonically with increasing storey-1 stiffness loss (mode 1: 1.56 Hz healthy to 1.30 Hz at 50% loss), providing the frequency-domain signal that the spectral classifiers ultimately exploit.",
];

// ---------------- Performance measures / comparison (paragraph split around Eq. 7)
const perfPara1a = "Table III reports the paired resource-level metrics that motivate this study's framing, together with an explicit accuracy-per-kilobyte efficiency ratio defined in (7),";
const perfPara1b = "By this measure the Proposed model achieves 10.8 accuracy points per kilobyte (int8), against 5.2 for the Full-signal baseline and 1.8 for Recon-GRU — roughly twice and six times more efficient, respectively — despite trailing on raw accuracy alone. The Proposed model is 2.3× smaller than the Full-signal baseline and 22.9× smaller than the Recon-GRU baseline in fp32, with a corresponding 131× lower CPU inference latency than Recon-GRU (0.21 ms versus 27.6 ms per window), the latter's bidirectional-GRU reconstruction stage being the dominant cost. The distinction matters beyond arithmetic: a Recon-GRU-class model, at 148.7 KB fp32 (45.3 KB int8), would consume most or all of the usable RAM on a typical Cortex-M0+-class microcontroller (commonly in the 32–64 KB range once the firmware and sensing stack are accounted for), whereas the Proposed model's 6.5–7.1 KB footprint leaves the overwhelming majority of such a budget free. Measured against the reconstruction-based approach this study most directly compares against " + cite(1) + ", the Proposed model therefore trades 4.2 points of average accuracy for close to two orders of magnitude less compute and memory, a trade that a genuinely constrained edge deployment would very plausibly accept, and one that several of the compared baselines may not even be in a position to offer, given their footprint.";
const perfPara2 = "A second, more modest but equally honest finding concerns quantisation. Dynamic int8 quantisation targets only Linear and GRU layers; Recon-GRU, being GRU-dominated, shrinks substantially (148.7 KB to 45.3 KB, a 3.3× reduction) with classification accuracy essentially unchanged (Table IV: mean differences of at most 0.3 percentage points, within the noise expected from a single run and not interpretable as a reliable improvement or degradation). The Proposed model, being convolution-dominated and already only 6.5 KB, sees no size benefit from this particular quantisation method (6.5 KB to 7.1 KB) and a small latency increase (0.21 ms to 0.38 ms) rather than a speed-up — consistent with the broader observation, also made in recent comparative TinyML compression work " + cite(10) + ", that pruning and quantisation are not interchangeable and depend on the target architecture. The practical implication is that a model already this compact is better served by fp32 deployment or by a static, graph-level quantisation method (e.g., TensorFlow Lite full-integer quantisation) than by PyTorch dynamic quantisation, which specifically targets Linear/RNN-heavy computation the Proposed architecture does not have much of.";

// ---------------- Ablation
const ablationParas = [
  "Fig. 3 isolates the contribution of self-supervised masked-spectrum pretraining by comparing the Proposed model against an identically sized encoder trained from random initialisation through the same corrupted-data fine-tuning stage. Pretraining is found to improve mean accuracy by 2.3 points under MCAR, 4.8 points under MAR, and 2.8 points under MNAR (3.3 points on average), a consistent, non-trivial effect given the modest size of both models. This result supports the design choice of separating representation learning, conducted on clean data where the reconstruction target is unambiguous, from the harder, corrupted-data classification objective, rather than attempting to learn both simultaneously from scratch.",
];

// ---------------- Limitations
const limitationsParas = [
  "This study carries four limitations that ought to be resolved before its findings are treated as field-validated. First, and most significant, the underlying dataset is a numerical simulation rather than recorded field data: the execution environment used here could not reach the standard recorded SHM archives (the LANL three-storey structure repository and the KW51 railway-bridge record on Zenodo, among others), so a physics-grounded but synthetic 4-DOF structure was used instead, with damage severities chosen to be separable (Fig. 4) rather than drawn from an unconstrained real damage distribution. Readers with normal internet access may substitute either real archive directly into the released code (Section III); this is, in the authors' assessment, the single most important next step. Second, all reported figures reflect a single random seed and a single training run per model, with no repeated-seed variance estimate; the differences discussed in Sections VI–VIII, particularly the sub-percentage-point quantisation deltas, should accordingly be read as indicative rather than statistically confirmed. Third, the edge-deployment claims rest on CPU latency and model size measured on a development machine rather than on an actual microcontroller (e.g., an ESP32 or Raspberry Pi Pico via TensorFlow Lite Micro); this is a reasonable proxy but not a substitute for on-device validation. Fourth, the missingness mechanisms considered here are simplified approximations of the taxonomy from the statistics literature " + cite(9) + ", and the damage scenario is restricted to a single-location stiffness reduction rather than the range a deployed system would need to generalise across; the domain-generalisation gap discussed in Section II remains open.",
];

// ---------------- Conclusion
const conclusionParas = [
  "This paper reframed missing- and noisy-sensor-data robustness in structural health monitoring around task-relevant accuracy per unit of resource, rather than signal-reconstruction fidelity under unconstrained compute, and quantified the resulting trade-off directly, including through an explicit accuracy-per-kilobyte efficiency measure. A compact, self-supervised-pretrained classifier gives up 4.2 accuracy points on average against a GRU-based reconstruction baseline drawn from recent literature, in exchange for a 42.9× reduction in parameters and a 131× reduction in CPU inference latency, is roughly twice to six times more efficient by the accuracy-per-kilobyte measure, and degrades more gracefully than a full-capacity baseline under the more realistic burst-dropout corruption mechanism, overtaking it beyond a 30% missingness rate. Self-supervised pretraining is shown to contribute a consistent, measurable accuracy gain, while post-training dynamic quantisation is found to help substantially for GRU-heavy models and not at all for an already-compact convolutional one, a distinction worth stating plainly rather than glossing over. The principal next step remains validating these findings on recorded field data rather than the numerical simulation used here; the complete, documented codebase released alongside this paper is intended to make that substitution direct.",
];

module.exports = {
  title, authors, affil, abstractText, keywords,
  introParas, litParas, contribBullets,
  resultsParas, ablationParas, limitationsParas, conclusionParas,
  cite, run, para, body, sectionHeading, subHeading, figureImage, makeTable, tableCaption,
};

// ================================================================== REFERENCES (IEEE numbered, 2024-2026, DOI-verified, numbered by first citation)
const references = [
  "H. Tran The, T. Bui Tien, and H. Tran Ngoc, \"Deep learning-based missing data reconstruction in structural health monitoring using gated dilated convolution and GRU,\" Int. J. Numer. Methods Eng., 2026, doi: 10.1002/nme.70257.",
  "G. Scarselli and F. Nicassio, \"Machine learning for structural health monitoring of aerospace structures: A review,\" Sensors, vol. 25, no. 19, p. 6136, 2025, doi: 10.3390/s25196136.",
  "S. A. Adimass and A. Żak, \"Review of digital twin in structural dynamics to improve condition monitoring,\" Struct. Control Health Monit., 2026, doi: 10.1155/stc/2777297.",
  "\"Trends and perspectives in structural health monitoring through edge computing: A review with zero-shot natural language processing categorization,\" J. Railway Sci. Technol., 2025, doi: 10.1016/j.jrst.2025.08.003.",
  "\"TinyML-enabled structural health monitoring for real-time anomaly detection in civil infrastructure,\" Alexandria Eng. J., 2025, doi: 10.1016/j.aej.2025.08.046.",
  "\"Secure edge computing reference architecture for data-driven structural health monitoring: Lessons learned from implementation and benchmarking,\" in Proc. 2025 ACM Southeast Conf. (ACM SE '25), 2025, doi: 10.1145/3696673.3723074.",
  "Y. Zhang, A. Nürnberg, L. S. Martinez Rau, Q. Nguyen Phuong Vu, Y. Lu, B. Oelmann, and S. Bader, \"TinyML pipeline for efficient crack classification in UAV-based structural health inspections,\" Sci. Rep., 2026, doi: 10.1038/s41598-026-43534-4.",
  "X. Jian, C. Stoura, S. Scandella, and E. Chatzi, \"Disentangling damage from operational variability: A label-free self-supervised representation learning framework for output-only structural damage identification,\" arXiv:2604.19658, 2026.",
  "C. Wongoutong, \"Performance comparison of various imputation methods for missing data mechanisms (MAR, MCAR, and MNAR) in a nonstationary time series,\" Int. J. Math. Math. Sci., 2025, doi: 10.1155/ijmm/3031708.",
  "M. Y. Shabir, G. Torta, and F. Damiani, \"TinyML model compression: A comparative study of pruning and quantization on selected standard and custom neural networks,\" Telecommun. Syst., vol. 88, no. 4, art. 132, 2025, doi: 10.1007/s11235-025-01363-2.",
  "X. Yu, H. Liu, J. Wang, X. Wen, Z. Ge, W. Chen, X. Fan, Z. Wang, and Z. Li, \"MAML bridges the data gap in deep learning-based structural health monitoring,\" Buildings, vol. 15, no. 17, p. 3163, 2025, doi: 10.3390/buildings15173163.",
  "A. I. F. Al-Adly and P. Kripakaran, \"Physics-informed neural networks for structural health monitoring: A case study for Kirchhoff-Love plates,\" Data-Centric Eng., vol. 5, p. e4, 2024, doi: 10.1017/dce.2024.4.",
  "H. Semmelrock, S. Kopeinik, D. Kowald, T. Ross-Hellauer, and E. Lex, \"Reproducibility in machine-learning-based research: Overview, barriers, and drivers,\" AI Magazine, 2025, doi: 10.1002/aaai.70002.",
];

// ================================================================== EQUATIONS (native OOXML/Word math objects)
const eq1 = eqBlock([
  mr("M"),
  mfrac([msup("d", "2"), mr("x")], [msup("dt", "2")]),
  mr(" + C"), mfrac([mr("dx")], [mr("dt")]),
  mr(" + Kx = F(t)"),
], "(1)");

const eq2 = eqBlock([mr("C = αM + βK")], "(2)");

const eq3 = eqBlock([
  msub("SNR", "dB"), mr(" = 10 "), msub("log", "10"),
  new MathRoundBrackets({ children: [mfrac([msub("P", "signal")], [msub("P", "noise")])] }),
], "(3)");

const eq4 = eqBlock([
  msub("f", "n"), mr(" = "), mfrac([mr("1")], [mr("2π")]),
  new MathRadical({ children: [mr("eig("), msup("M", "-1"), mr("K)")] }),
], "(4)");

const eq5 = eqBlock([mr("Accuracy = "), mfrac([mr("TP + TN")], [mr("TP + TN + FP + FN")])], "(5)");
const eq6 = eqBlock([mr("F1 = "), mfrac([mr("2PR")], [mr("P + R")])], "(6)");
const eq7 = eqBlock([mr("Efficiency = "), mfrac([mr("Accuracy (%)")], [mr("Size (KB)")])], "(7)");

// ================================================================== FIGURE PIXEL DIMENSIONS (single-column IEEE width)
function pngDims(path) {
  const buf = fs.readFileSync(path);
  const w = buf.readUInt32BE(16);
  const h = buf.readUInt32BE(20);
  return { w, h };
}
function scaledTo(path, targetW) {
  const d = pngDims(path);
  return { w: targetW, h: Math.round(targetW * d.h / d.w) };
}

const fig1Dim = scaledTo(`${OUT}/fig1_accuracy_vs_missingness.png`, 195);
const fig2Dim = scaledTo(`${OUT}/fig2_accuracy_vs_resource.png`, 220);
const fig3Dim = scaledTo(`${OUT}/fig3_ablation_pretraining.png`, 220);
const fig4Dim = scaledTo(`${OUT}/fig4_modal_frequency_shift.png`, 220);

// ================================================================== TABLES
const tableI = makeTable(
  ["Item", "Value"],
  [
    ["Structure model", "4-DOF shear building (lumped mass)"],
    ["Sampling rate / duration", "200 Hz / 20 s per run"],
    ["Channels", "4 (one accelerometer per floor)"],
    ["Runs (healthy / damaged)", "40 / 40 (80 total)"],
    ["Damage severities", "20%, 35%, 50% story-1 stiffness loss"],
    ["Measurement noise", "Additive Gaussian, 15 dB SNR"],
    ["Train / val / test split (runs)", "56 / 12 / 12 (run-level, no leakage)"],
    ["Window / stride", "256 samples (1.28 s) / 128 samples"],
    ["Windows (train / val / test)", "1,680 / 360 / 360"],
  ],
  [1900, 2900]
);

function pct(x) { return (x * 100).toFixed(1); }
const mainResults = {
  "Full-signal baseline": { c: [0.800462962962963, 0.8757537278593315, 0.8142181069958848], clean: [0.8638888888888889, 0.9070208728652752, 0.923744855967078] },
  "Naive-impute baseline": { c: [0.8268518518518518, 0.888103947070607, 0.8668106995884773], clean: [0.8638888888888889, 0.9070208728652752, 0.923744855967078] },
  "Recon-GRU baseline": { c: [0.8113425925925926, 0.8736105582721766, 0.8560596707818929], clean: [0.8444444444444444, 0.8935361216730038, 0.9024691358024692] },
  "Proposed - no pretrain": { c: [0.7363425925925925, 0.8058223594304804, 0.8219821673525377], clean: [0.775, 0.8316008316008316, 0.8703703703703705] },
  "Proposed (ours)": { c: [0.7685185185185185, 0.8327660982548936, 0.8446879286694102], clean: [0.7916666666666666, 0.8466257668711656, 0.8730041152263374] },
};
const tableII = makeTable(
  ["Model", "Acc. (%)", "F1 (%)", "AUC (%)", "Clean Acc. (%)"],
  Object.entries(mainResults).map(([name, v]) => [name, pct(v.c[0]), pct(v.c[1]), pct(v.c[2]), pct(v.clean[0])]),
  [1660, 800, 800, 800, 800]
);

// Table III: resource profile (fp32 vs int8) + accuracy-per-KB efficiency ratio (Eq. 7)
const tableIII = makeTable(
  ["Model", "Params", "fp32 (KB)", "int8 (KB)", "Lat. fp32 (ms)", "Lat. int8 (ms)", "Eff. (pts/KB)"],
  [
    ["Full-signal (CNN)", "2,994", "14.8", "15.3", "0.24", "0.32", "5.2"],
    ["Recon-GRU", "36,772", "148.7", "45.3", "27.58", "25.60", "1.8"],
    ["Proposed (ours)", "858", "6.5", "7.1", "0.21", "0.38", "10.8"],
  ],
  [950, 620, 580, 580, 630, 630, 700]
);

// Mean over the two rates evaluated (20%, 30%) per mechanism -- see perfPara2.
const tableIV = makeTable(
  ["Corruption", "Acc. fp32 (%)", "Acc. int8 (%)", "Δ (pp)"],
  [
    ["MCAR", "78.2", "78.3", "+0.1"],
    ["MAR", "78.2", "78.5", "+0.3"],
    ["MNAR", "74.2", "74.4", "+0.3"],
  ],
  [1300, 1000, 1000, 800]
);

// ================================================================== DOCUMENT ASSEMBLY
function refPara(n, text) {
  return new Paragraph({
    children: [run(`[${n}] `, {}), run(text, {})],
    spacing: { after: 60 },
    indent: { left: 200, hanging: 200 },
    alignment: AlignmentType.JUSTIFIED,
  });
}

const PAGE = {
  size: { width: 12240, height: 15840 }, // US Letter, twips (8.5in x 11in)
  margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 }, // 0.75in
};

const titleSection = {
  properties: { type: SectionType.CONTINUOUS, page: PAGE },
  children: [
    new Paragraph({
      children: [run(title, { bold: true, size: 30 })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 160 },
    }),
    new Paragraph({
      children: [run(authors, { size: 24 })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 40 },
    }),
    new Paragraph({
      children: [run(affil, { size: 20, italics: true })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    }),
  ],
};

const bodyChildren = [];

bodyChildren.push(
  para([run("Abstract—", { bold: true, italics: true }), run(abstractText)], { spacing: { after: 100 } }),
  para([run("Index Terms—", { bold: true, italics: true }), run(keywords)], { spacing: { after: 160 } })
);

// I. Introduction
bodyChildren.push(sectionHeading("I", "Introduction"));
introParas.forEach(t => bodyChildren.push(body(t)));

// II. Literature Review
bodyChildren.push(sectionHeading("II", "Literature Review and Comparison with Existing Methods"));
litParas.forEach(t => bodyChildren.push(body(t)));

// III. Contribution
bodyChildren.push(sectionHeading("III", "Contribution"));
contribBullets.forEach(t => bodyChildren.push(para([run("• "), run(t)], { indent: { left: 200 } })));

// IV. Methodology (text interleaved with native equations)
bodyChildren.push(sectionHeading("IV", "Methodology"));

bodyChildren.push(subHeading("A. Simulated structure and dataset."));
bodyChildren.push(body("Direct network access to standard recorded SHM archives — the LANL three-storey structure repository and the KW51 bridge-monitoring record on Zenodo, among others — was not available from this study's execution environment; Section IX discusses this constraint and its implications candidly. In its place, a 4-DOF lumped-mass shear-building structure is simulated, governed by the equation of motion given in (1),"));
bodyChildren.push(eq1);
bodyChildren.push(body("where M = diag(m₁, m₂, m₃, m₄) is the storey-mass matrix, K is assembled from the storey stiffnesses k₁,…,k₄ following the standard shear-building convention, and the damping matrix C follows the Rayleigh form of (2),"));
bodyChildren.push(eq2);
bodyChildren.push(body("Each simulated run applies band-limited random floor excitation for 20 s at 200 Hz, producing a four-channel (one accelerometer per floor) acceleration time-series with additive Gaussian measurement noise fixed at a signal-to-noise ratio of 15 dB, as defined in (3),"));
bodyChildren.push(eq3);
bodyChildren.push(body("The healthy class retains nominal storey stiffness; the damaged class reduces storey-1 stiffness by 20%, 35%, or 50%, severities chosen — via the natural-frequency separability relation of (4) and the check reported in Fig. 4 — to be detectable yet realistic for a low-order structure."));
bodyChildren.push(eq4);
bodyChildren.push(body("The dataset comprises 80 independent runs (40 healthy, 40 damaged), split at the run level, not the window level, to avoid leakage, into 56 training, 12 validation, and 12 test runs, windowed into 256-sample (1.28 s) segments with 128-sample stride, giving 1,680 training, 360 validation, and 360 test windows (Table I)."));

bodyChildren.push(subHeading("B. Missingness and noise injection."));
bodyChildren.push(body("Three canonical missing-data mechanisms are simulated, applied to raw time-domain windows, following the taxonomy examined in recent comparative imputation work " + cite(9) + ". MCAR (missing completely at random) zeros individual samples independently per channel. MAR is approximated as burst dropout, zeroing contiguous 10-sample blocks to emulate a sensor or link outage — arguably the more realistic field failure mode. MNAR is approximated as amplitude-dependent dropout, in which the largest-magnitude samples per channel are zeroed, emulating saturation during high-amplitude, potentially damage-relevant transients, the condition existing imputation work stress-tests least. Each mechanism is applied at rates of 10%, 20%, 30%, and 40%."));

bodyChildren.push(subHeading("C. Spectral feature representation."));
bodyChildren.push(body("All classifiers operate on a log-magnitude spectrum of the (possibly corrupted) time-domain window, restricted to 0–20 Hz, the band covering the structure's first four elastic modes, rather than on raw time samples. This mirrors standard ambient-vibration SHM practice, which band-limits to the known modal range, and proved necessary in practice: a preliminary check found the informative low-frequency content was diluted to the point of non-learnability when a compact model pooled over the full-band raw time series (Section IX). The reconstruction baseline described below is the only model that reconstructs a time-domain waveform, since that is the literal task of the method it represents; its output is transformed to the same spectral representation before classification, keeping all four models' classification stage directly comparable."));

bodyChildren.push(subHeading("D. Models compared."));
bodyChildren.push(body("(i) Full-signal baseline: a 1D-CNN (two convolution-ReLU-pool blocks, global average pooling, linear head; 2,994 parameters) trained on clean spectra only and evaluated as-is on corrupted spectra. (ii) Naive-impute baseline: the same trained classifier, preceded by per-channel linear interpolation of corrupted time samples before the spectral transform. (iii) Recon-GRU baseline: a bidirectional-GRU encoder/GRU decoder denoising autoencoder (36,772 parameters) that reconstructs the clean time-domain window from corrupted input, approximating the gated-dilated-convolution-plus-GRU approach of " + cite(1) + "; its reconstruction is classified by model (i)'s trained head. (iv) Proposed: a compact encoder (two convolution layers of 8 and 16 channels, a single max-pool; 858 parameters in total with its classification head) applied directly to the corrupted spectrum, with no explicit reconstruction step at inference."));

bodyChildren.push(subHeading("E. Self-supervised pretraining."));
bodyChildren.push(body("The proposed encoder is first pretrained through masked-spectrum reconstruction: random frequency bands of the clean training spectra are zeroed, and a lightweight decoder (discarded after pretraining) is trained to reconstruct the full clean spectrum — a spectral analogue of the masked-signal pretraining used elsewhere in self-supervised sequence learning, and broadly consistent in spirit with recent label-free representation-learning approaches proposed specifically for SHM " + cite(8) + ", though the present formulation operates in the frequency domain rather than on output-only modal features. The pretrained encoder is then fine-tuned with a classification head directly on spectra corrupted by a random mixture of MCAR/MAR/MNAR at random rates in [10%, 40%], so that the model is exposed during training to the actual corruption distribution it will face at test time. An ablation variant, referred to as \"no pretrain\", uses an identically sized, randomly initialised encoder taken through the same corrupted-data fine-tuning stage, isolating the specific contribution of the pretraining step."));

bodyChildren.push(subHeading("F. Quantisation."));
bodyChildren.push(body("The trained Recon-GRU and Proposed models are additionally evaluated after PyTorch post-training dynamic quantisation (int8, applied to Linear and GRU layers), measuring model size (serialised state dictionary) and CPU inference latency (mean over 200 repeated forward passes after a 10-pass warm-up) before and after quantisation. This mirrors, on a smaller scale, the spirit of recent comparative TinyML compression studies " + cite(10) + ", though the present work restricts itself to a single, widely available compression method applied consistently across models rather than a broader method sweep."));

bodyChildren.push(tableCaption("TABLE I", "Simulated Dataset Summary"));
bodyChildren.push(tableI);
bodyChildren.push(new Paragraph({ text: "", spacing: { after: 90 } }));
bodyChildren.push(...figureImage(`${OUT}/fig4_modal_frequency_shift.png`, fig4Dim,
  "Simulated natural-frequency shift with increasing story-1 stiffness loss, confirming the damage manipulation is physically grounded.", "Fig. 4"));

// V. Experimental Setup
bodyChildren.push(sectionHeading("V", "Experimental Setup"));
bodyChildren.push(body(setupPara1));
bodyChildren.push(eq5);
bodyChildren.push(eq6);
bodyChildren.push(body(setupPara2));

// VI. Results
bodyChildren.push(sectionHeading("VI", "Results"));
resultsParas.forEach(t => bodyChildren.push(body(t)));
bodyChildren.push(tableCaption("TABLE II", "Classification Accuracy, F1 and AUC (mean over 12 corrupted test conditions, and on clean data)"));
bodyChildren.push(tableII);
bodyChildren.push(new Paragraph({ text: "", spacing: { after: 90 } }));
bodyChildren.push(...figureImage(`${OUT}/fig1_accuracy_vs_missingness.png`, fig1Dim,
  "Accuracy vs. corruption rate under MCAR, MAR (burst dropout) and MNAR (amplitude clipping) missingness.", "Fig. 1"));

// VII. Performance Measures and Comparison with Existing Methods
bodyChildren.push(sectionHeading("VII", "Performance Measures and Resource Comparison"));
bodyChildren.push(body(perfPara1a));
bodyChildren.push(eq7);
bodyChildren.push(body(perfPara1b));
bodyChildren.push(tableCaption("TABLE III", "Model Size, CPU Inference Latency (fp32 vs. int8) and Accuracy-per-Kilobyte Efficiency"));
bodyChildren.push(tableIII);
bodyChildren.push(new Paragraph({ text: "", spacing: { after: 90 } }));
bodyChildren.push(body(perfPara2));
bodyChildren.push(tableCaption("TABLE IV", "Proposed Model: Classification Accuracy Before/After int8 Quantisation, by Corruption Condition"));
bodyChildren.push(tableIV);
bodyChildren.push(new Paragraph({ text: "", spacing: { after: 90 } }));
bodyChildren.push(...figureImage(`${OUT}/fig2_accuracy_vs_resource.png`, fig2Dim,
  "Mean accuracy across all corrupted conditions vs. quantized (int8) model size (log scale).", "Fig. 2"));

// VIII. Ablation Study
bodyChildren.push(sectionHeading("VIII", "Ablation Study"));
ablationParas.forEach(t => bodyChildren.push(body(t)));
bodyChildren.push(...figureImage(`${OUT}/fig3_ablation_pretraining.png`, fig3Dim,
  "Effect of self-supervised masked-spectrum pretraining on mean accuracy, by corruption mechanism.", "Fig. 3"));

// IX. Limitations
bodyChildren.push(sectionHeading("IX", "Limitations"));
limitationsParas.forEach(t => bodyChildren.push(body(t)));

// X. Conclusion
bodyChildren.push(sectionHeading("X", "Conclusion"));
conclusionParas.forEach(t => bodyChildren.push(body(t)));

// References
bodyChildren.push(new Paragraph({
  children: [run("References", { bold: true, size: 21 })],
  alignment: AlignmentType.CENTER,
  spacing: { before: 200, after: 100 },
}));
references.forEach((r, i) => bodyChildren.push(refPara(i + 1, r)));

const bodySection = {
  properties: {
    type: SectionType.CONTINUOUS,
    page: PAGE,
    column: { count: 2, space: 360, equalWidth: true },
  },
  children: bodyChildren,
};

const doc = new Document({
  sections: [titleSection, bodySection],
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("SHM_Edge_Robustness_IEEE_Paper.docx", buf);
  console.log("Wrote SHM_Edge_Robustness_IEEE_Paper.docx");
});
