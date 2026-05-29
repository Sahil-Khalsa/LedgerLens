import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        brand: { 50: "#f0f9ff", 500: "#0ea5e9", 900: "#0c4a6e" },
      },
    },
  },
  plugins: [],
};

export default config;
