import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Enso Atlas - Pathology Evidence Engine",
  description:
    "On-premise pathology evidence engine for treatment-response insight. Provides interpretable AI predictions with evidence-based heatmaps and structured reports.",
  keywords: [
    "pathology",
    "AI",
    "digital pathology",
    "whole slide imaging",
    "treatment response",
    "medical imaging",
  ],
  authors: [{ name: "Enso Labs" }],
  robots: "noindex, nofollow", // Research tool - not for public indexing
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        {/* OpenSeadragon images will be loaded from CDN */}
        <link
          rel="preconnect"
          href="https://cdnjs.cloudflare.com"
          crossOrigin="anonymous"
        />
      </head>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
