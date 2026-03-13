import './globals.css'
import type { Metadata } from 'next'
import { Providers } from './Providers'

export const metadata: Metadata = {
  title: 'SloughGPT - Enterprise AI',
  description: 'Enterprise-grade AI framework with production-ready ML infrastructure',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
