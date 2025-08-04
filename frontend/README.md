# QuantumFlow AI Lab - Frontend

This is the frontend dashboard for QuantumFlow AI Lab, an AI-powered trading platform built with [Next.js](https://nextjs.org), TypeScript, and shadcn/ui components.

## 🚀 Features

- **Real-time Market Data**: Live stock prices from Marketstack API
- **AI-Powered Insights**: Intelligent market analysis and strategy generation
- **Modern UI**: Built with shadcn/ui components and Tailwind CSS
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Dark Theme**: Beautiful gradient background with glassmorphism effects

## 🛠️ Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: shadcn/ui
- **Icons**: Lucide React
- **Backend Integration**: REST API + WebSocket support

## Getting Started

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

3. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🔗 Backend Integration

The frontend automatically connects to the QuantumFlow AI Lab backend:
- **API Endpoint**: `http://localhost:8001`
- **Market Data**: `/api/market/data`
- **Stock Data**: `/api/stocks/{symbol}`
- **WebSocket**: `ws://localhost:8001/ws/demo`

Make sure the backend server is running on port 8001 for full functionality.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
