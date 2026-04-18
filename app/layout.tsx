import type { Metadata, Viewport } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Bangers } from "next/font/google";
import "./globals.css";

const bangers = Bangers({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-comic",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Omni ✿",
  description: "Tap anything. It grows a face and talks back.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#ffd6f0",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="zh-CN"
      className={`${GeistSans.variable} ${GeistMono.variable} ${bangers.variable}`}
    >
      <head>
        {/* Bounce insecure LAN loads up to HTTPS so navigator.mediaDevices
            (camera/mic) is available. Runs before React hydrates. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var l=window.location;if(l.protocol==='http:'&&l.hostname!=='localhost'&&l.hostname!=='127.0.0.1'&&!/^\\[?::1\\]?$/.test(l.hostname)){l.replace('https://'+l.host+l.pathname+l.search+l.hash);}}catch(e){}})();`,
          }}
        />
      </head>
      <body className="font-sans text-[#2a1540] antialiased">
        {children}
      </body>
    </html>
  );
}
