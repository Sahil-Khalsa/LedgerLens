import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "LedgerLens — SEC Filing Intelligence",
  description: "Verified financial answers from SEC filings, cited to the source page.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-gray-50 text-gray-900 min-h-screen antialiased font-[var(--font-inter)]">
        {children}
      </body>
    </html>
  );
}
