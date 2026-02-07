import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Clinical color palette - professional medical software aesthetic
        clinical: {
          50: "#f0f9ff",
          100: "#e0f2fe",
          200: "#bae6fd",
          300: "#7dd3fc",
          400: "#38bdf8",
          500: "#0ea5e9",
          600: "#0284c7",
          700: "#0369a1",
          800: "#075985",
          900: "#0c4a6e",
          950: "#082f49",
        },
        // Status colors for medical contexts - high contrast, accessible
        status: {
          positive: "#059669",
          "positive-light": "#d1fae5",
          negative: "#dc2626",
          "negative-light": "#fee2e2",
          neutral: "#6b7280",
          "neutral-light": "#f3f4f6",
          warning: "#d97706",
          "warning-light": "#fef3c7",
        },
        // Responder/Non-responder specific colors
        responder: {
          positive: "#047857",
          "positive-bg": "#ecfdf5",
          "positive-border": "#a7f3d0",
          negative: "#b91c1c",
          "negative-bg": "#fef2f2",
          "negative-border": "#fecaca",
        },
        // Surface colors for panels - subtle depth hierarchy
        surface: {
          primary: "#ffffff",
          secondary: "#f8fafc",
          tertiary: "#f1f5f9",
          elevated: "#ffffff",
          border: "#e2e8f0",
          "border-strong": "#cbd5e1",
        },
        // Navy for professional headers
        navy: {
          50: "#f8fafc",
          100: "#f1f5f9",
          600: "#475569",
          700: "#334155",
          800: "#1e293b",
          900: "#0f172a",
          950: "#020617",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "SF Mono", "Menlo", "monospace"],
        display: ["Inter", "system-ui", "sans-serif"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],
      },
      boxShadow: {
        clinical: "0 1px 3px 0 rgb(0 0 0 / 0.05), 0 1px 2px -1px rgb(0 0 0 / 0.05)",
        "clinical-md": "0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05)",
        "clinical-lg": "0 10px 15px -3px rgb(0 0 0 / 0.05), 0 4px 6px -4px rgb(0 0 0 / 0.05)",
        "clinical-xl": "0 20px 25px -5px rgb(0 0 0 / 0.05), 0 8px 10px -6px rgb(0 0 0 / 0.05)",
        inner: "inset 0 2px 4px 0 rgb(0 0 0 / 0.05)",
        glow: "0 0 20px rgb(14 165 233 / 0.15)",
      },
      borderRadius: {
        clinical: "0.5rem",
      },
      animation: {
        "fade-in": "fadeIn 0.2s ease-out",
        "fade-in-up": "fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
        "slide-up": "slideUp 0.3s ease-out",
        "slide-in-right": "slideInRight 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
        "pulse-subtle": "pulseSubtle 2s ease-in-out infinite",
        "scale-in": "scaleIn 0.25s cubic-bezier(0.34, 1.56, 0.64, 1)",
        "shimmer": "shimmer 2s linear infinite",
        "bounce-subtle": "bounceSubtle 0.6s ease-out",
        "glow": "glow 2s ease-in-out infinite",
        "slide-down": "slideDown 0.3s ease-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        fadeInUp: {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideInRight: {
          "0%": { opacity: "0", transform: "translateX(24px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        scaleIn: {
          "0%": { opacity: "0", transform: "scale(0.9)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        pulseSubtle: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.7" },
        },
        shimmer: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        bounceSubtle: {
          "0%": { transform: "scale(0.95)" },
          "50%": { transform: "scale(1.02)" },
          "100%": { transform: "scale(1)" },
        },
        glow: {
          "0%, 100%": { boxShadow: "0 0 20px rgba(14, 165, 233, 0.2)" },
          "50%": { boxShadow: "0 0 30px rgba(14, 165, 233, 0.4)" },
        },
        slideDown: {
          "0%": { opacity: "0", transform: "translateY(-100%)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      transitionTimingFunction: {
        "out-expo": "cubic-bezier(0.16, 1, 0.3, 1)",
        "out-back": "cubic-bezier(0.34, 1.56, 0.64, 1)",
      },
    },
  },
  plugins: [],
};

export default config;
