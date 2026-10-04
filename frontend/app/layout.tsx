import type { Metadata } from "next";
import "./styles.css";
export const metadata: Metadata = { title: "FPL AI | Football analytics", description: "Fantasy Premier League research, historical model results and squad tools." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
