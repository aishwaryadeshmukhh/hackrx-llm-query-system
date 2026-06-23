import './globals.css'

export const metadata = {
  title: 'PolicyMind — Insurance Policy Analyzer',
  description: 'Ask any coverage question. Get a clause-cited answer in seconds.',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
