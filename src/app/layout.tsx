import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Learn — How an LLM is trained",
  description:
    "A visual, step-by-step guide to how a large language model is trained — from raw text to a talking model.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
