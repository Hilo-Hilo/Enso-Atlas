// Enso Atlas - Client-side PDF Export
// Uses jsPDF to generate professional clinical reports

import jsPDF from "jspdf";
import type { StructuredReport } from "@/types";
import type { CaseNote } from "@/components/panels/CaseNotesPanel";

interface PdfExportOptions {
  report: StructuredReport;
  slideId: string;
  caseNotes?: CaseNote[];
  institutionName?: string;
}

/**
 * Generate a professional PDF report from analysis results
 */
export async function generatePdfReport(options: PdfExportOptions): Promise<Blob> {
  const { report, slideId, caseNotes = [], institutionName = "Enso Labs" } = options;

  const doc = new jsPDF({
    orientation: "portrait",
    unit: "mm",
    format: "a4",
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 20;
  const contentWidth = pageWidth - 2 * margin;
  let yPos = margin;

  // Helper: Add page break if needed
  const checkPageBreak = (requiredSpace: number) => {
    if (yPos + requiredSpace > pageHeight - margin) {
      doc.addPage();
      yPos = margin;
      return true;
    }
    return false;
  };

  // Helper: Draw section header
  const drawSectionHeader = (title: string, color: [number, number, number] = [0, 102, 153]) => {
    checkPageBreak(15);
    doc.setFillColor(...color);
    doc.rect(margin, yPos, contentWidth, 8, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text(title, margin + 3, yPos + 5.5);
    doc.setTextColor(0, 0, 0);
    yPos += 12;
  };

  // Helper: Add wrapped text
  const addWrappedText = (text: string, fontSize: number = 10, indent: number = 0) => {
    doc.setFontSize(fontSize);
    doc.setFont("helvetica", "normal");
    const lines = doc.splitTextToSize(text, contentWidth - indent);
    lines.forEach((line: string) => {
      checkPageBreak(6);
      doc.text(line, margin + indent, yPos);
      yPos += fontSize * 0.4 + 1;
    });
  };

  // ========== HEADER ==========
  doc.setFillColor(0, 102, 153);
  doc.rect(0, 0, pageWidth, 35, "F");

  doc.setTextColor(255, 255, 255);
  doc.setFontSize(20);
  doc.setFont("helvetica", "bold");
  doc.text("Pathology Analysis Report", margin, 18);

  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.text(`${institutionName} | AI-Assisted Diagnostic Platform`, margin, 26);

  yPos = 45;

  // ========== CASE INFORMATION ==========
  doc.setTextColor(0, 0, 0);
  doc.setFillColor(245, 245, 245);
  doc.rect(margin, yPos, contentWidth, 20, "F");

  doc.setFontSize(9);
  doc.setFont("helvetica", "bold");
  doc.text("Case ID:", margin + 3, yPos + 6);
  doc.text("Slide ID:", margin + 3, yPos + 12);
  doc.text("Generated:", margin + 3, yPos + 18);

  doc.setFont("helvetica", "normal");
  doc.text(report.caseId || "N/A", margin + 30, yPos + 6);
  doc.text(slideId, margin + 30, yPos + 12);
  doc.text(new Date(report.generatedAt).toLocaleString(), margin + 30, yPos + 18);

  // Task info on right side
  doc.setFont("helvetica", "bold");
  doc.text("Task:", margin + 90, yPos + 6);
  doc.setFont("helvetica", "normal");
  doc.text(report.task || "Treatment Response Prediction", margin + 105, yPos + 6);

  yPos += 28;

  // ========== MODEL OUTPUT ==========
  drawSectionHeader("Model Prediction");

  const prediction = report.modelOutput;
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");

  // Color based on prediction
  if (prediction.label.toLowerCase().includes("responder") && !prediction.label.toLowerCase().includes("non")) {
    doc.setTextColor(34, 139, 34);
  } else {
    doc.setTextColor(178, 34, 34);
  }
  doc.text(prediction.label, margin, yPos);
  doc.setTextColor(0, 0, 0);

  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.text(`Confidence: ${Math.round(prediction.confidence * 100)}%`, margin + 60, yPos);
  doc.text(`Score: ${prediction.score.toFixed(4)}`, margin + 110, yPos);
  yPos += 8;

  if (prediction.calibrationNote) {
    doc.setFontSize(8);
    doc.setTextColor(100, 100, 100);
    doc.text(`Note: ${prediction.calibrationNote}`, margin, yPos);
    doc.setTextColor(0, 0, 0);
    yPos += 6;
  }
  yPos += 4;

  // ========== CLINICAL SUMMARY ==========
  drawSectionHeader("Clinical Summary");
  addWrappedText(report.summary, 10);
  yPos += 6;

  // ========== SUPPORTING EVIDENCE ==========
  if (report.evidence.length > 0) {
    drawSectionHeader("Supporting Evidence");

    report.evidence.forEach((item, index) => {
      checkPageBreak(25);

      doc.setFontSize(10);
      doc.setFont("helvetica", "bold");
      doc.text(`Evidence #${index + 1}`, margin, yPos);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(100, 100, 100);
      doc.text(`Coordinates: (${item.coordsLevel0[0].toLocaleString()}, ${item.coordsLevel0[1].toLocaleString()})`, margin + 30, yPos);
      doc.setTextColor(0, 0, 0);
      yPos += 5;

      addWrappedText(item.morphologyDescription, 9, 3);

      doc.setFontSize(9);
      doc.setTextColor(0, 102, 153);
      addWrappedText(`Significance: ${item.whyThisPatchMatters}`, 9, 3);
      doc.setTextColor(0, 0, 0);
      yPos += 4;
    });
    yPos += 2;
  }

  // ========== SIMILAR CASES ==========
  if (report.similarExamples.length > 0) {
    drawSectionHeader("Reference Cohort Comparison");

    // Table header
    doc.setFillColor(230, 230, 230);
    doc.rect(margin, yPos, contentWidth, 6, "F");
    doc.setFontSize(9);
    doc.setFont("helvetica", "bold");
    doc.text("Case ID", margin + 3, yPos + 4);
    doc.text("Outcome", margin + 70, yPos + 4);
    doc.text("Distance", margin + 120, yPos + 4);
    yPos += 8;

    doc.setFont("helvetica", "normal");
    report.similarExamples.forEach((example) => {
      checkPageBreak(6);
      doc.text(example.exampleId.slice(0, 24), margin + 3, yPos);
      doc.text(example.label || "N/A", margin + 70, yPos);
      doc.text(example.distance.toFixed(4), margin + 120, yPos);
      yPos += 5;
    });
    yPos += 4;
  }

  // ========== CASE NOTES ==========
  if (caseNotes.length > 0) {
    drawSectionHeader("Clinical Notes", [75, 75, 75]);

    caseNotes.forEach((note) => {
      checkPageBreak(15);

      doc.setFontSize(8);
      doc.setFont("helvetica", "bold");
      const categoryLabel = note.category ? `[${note.category.toUpperCase()}] ` : "";
      doc.text(`${categoryLabel}${note.author} - ${new Date(note.createdAt).toLocaleString()}`, margin, yPos);
      yPos += 4;

      doc.setFont("helvetica", "normal");
      addWrappedText(note.content, 9, 2);
      yPos += 3;
    });
  }

  // ========== SUGGESTED NEXT STEPS ==========
  if (report.suggestedNextSteps.length > 0) {
    drawSectionHeader("Suggested Next Steps", [0, 102, 204]);

    report.suggestedNextSteps.forEach((step, index) => {
      checkPageBreak(8);
      doc.setFontSize(10);
      doc.setFont("helvetica", "normal");
      const stepText = `${index + 1}. ${step}`;
      addWrappedText(stepText, 10, 3);
    });
    yPos += 2;
  }

  // ========== LIMITATIONS ==========
  if (report.limitations.length > 0) {
    drawSectionHeader("Limitations", [204, 153, 0]);

    report.limitations.forEach((limitation) => {
      checkPageBreak(8);
      addWrappedText(`- ${limitation}`, 9, 3);
    });
    yPos += 2;
  }

  // ========== SAFETY STATEMENT ==========
  checkPageBreak(30);
  doc.setFillColor(255, 235, 235);
  doc.setDrawColor(204, 0, 0);
  doc.setLineWidth(0.5);

  const safetyLines = doc.splitTextToSize(report.safetyStatement, contentWidth - 10);
  const safetyHeight = safetyLines.length * 4 + 14;

  doc.roundedRect(margin, yPos, contentWidth, safetyHeight, 2, 2, "FD");

  doc.setTextColor(153, 0, 0);
  doc.setFontSize(10);
  doc.setFont("helvetica", "bold");
  doc.text("IMPORTANT SAFETY NOTICE", margin + 5, yPos + 6);

  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  safetyLines.forEach((line: string, idx: number) => {
    doc.text(line, margin + 5, yPos + 12 + idx * 4);
  });

  doc.setTextColor(0, 0, 0);

  // ========== FOOTER ==========
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(128, 128, 128);
    doc.text(
      `Page ${i} of ${totalPages} | Generated by Enso Atlas v0.1.0 | For Research Use Only`,
      pageWidth / 2,
      pageHeight - 10,
      { align: "center" }
    );
  }

  return doc.output("blob");
}

/**
 * Download the generated PDF
 */
export function downloadPdf(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
