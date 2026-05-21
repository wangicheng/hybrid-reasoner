import { useState, useRef, useEffect } from "react";
import { cn } from "../lib/utils";

export function VanishInput({
  placeholders = ["Search for light novels..."],
  onChange,
  onSubmit,
  className,
}) {
  const [currentPlaceholder, setCurrentPlaceholder] = useState(0);
  const [value, setValue] = useState("");
  const [isVanishing, setIsVanishing] = useState(false);
  const inputRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    let interval;
    if (!isVanishing) {
      interval = setInterval(() => {
        setCurrentPlaceholder((prev) => (prev + 1) % placeholders.length);
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [placeholders.length, isVanishing]);

  const drawTextOnCanvas = (text) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    // Scale for high DPI
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Match input styles
    const inputStyle = window.getComputedStyle(inputRef.current);
    ctx.font = `${inputStyle.fontWeight} ${inputStyle.fontSize} ${inputStyle.fontFamily}`;
    ctx.fillStyle = inputStyle.color;
    ctx.textBaseline = "middle";

    // Get padding to match text position
    const paddingLeft = parseFloat(inputStyle.paddingLeft);
    ctx.fillText(text, paddingLeft, rect.height / 2);

    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  };

  const createParticles = (imageData, rect) => {
    const particles = [];
    const { data, width, height } = imageData;
    const dpr = window.devicePixelRatio || 1;

    // Scan the image data for non-transparent pixels
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        const i = (y * width + x) * 4;
        const alpha = data[i + 3];

        if (alpha > 128) { // If pixel is mostly opaque
          particles.push({
            x: x / dpr,
            y: y / dpr,
            vx: (Math.random() - 0.5) * 8 + (Math.random() > 0.5 ? 2 : -2), // Move outward/upward
            vy: (Math.random() - 1) * 5 - 2,
            size: Math.random() * 2 + 1,
            life: 1,
            decay: Math.random() * 0.02 + 0.02,
            color: `rgba(${data[i]}, ${data[i + 1]}, ${data[i + 2]}, 1)`
          });
        }
      }
    }
    return particles;
  };

  const triggerVanish = (e) => {
    e.preventDefault();
    if (isVanishing || !value) return;

    setIsVanishing(true);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();

    // Get text pixels
    const imageData = drawTextOnCanvas(value);
    if (!imageData) {
      finishSubmit(e);
      return;
    }

    let particles = createParticles(imageData, rect);

    // Animate particles
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      let alive = false;
      particles.forEach(p => {
        if (p.life > 0) {
          alive = true;
          p.x += p.vx;
          p.y += p.vy;
          p.life -= p.decay;

          ctx.globalAlpha = Math.max(0, p.life);
          ctx.fillStyle = "#00f0ff"; // Glow color override or use p.color
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      if (alive) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        finishSubmit(e);
      }
    };

    animate();
  };

  const finishSubmit = (e) => {
    setIsVanishing(false);
    if (onSubmit) onSubmit(value);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      triggerVanish(e);
    }
  };

  return (
    <div className={cn("relative w-full max-w-xl mx-auto h-14", className)}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 z-20 pointer-events-none w-full h-full"
      />
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => {
          if (!isVanishing) {
            setValue(e.target.value);
            if (onChange) onChange(e);
          }
        }}
        onKeyDown={handleKeyDown}
        className={cn(
          "w-full h-full bg-zinc-900/50 border border-zinc-800 rounded-full px-6 text-zinc-100 text-lg shadow-[0_0_20px_rgba(0,0,0,0.5)] focus:outline-none focus:border-purple-500/50 focus:ring-1 focus:ring-purple-500/50 transition-all duration-300 placeholder:text-transparent z-10 relative",
          isVanishing && "text-transparent"
        )}
      />

      {!value && !isVanishing && (
        <div className="absolute inset-y-0 left-6 flex items-center pointer-events-none">
          <span className="text-zinc-500 text-lg font-medium animate-pulse">
            {placeholders[currentPlaceholder]}
          </span>
        </div>
      )}

      {/* Glow effect behind input */}
      <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500/20 via-purple-500/20 to-pink-500/20 rounded-full blur-md -z-10 opacity-50 group-hover:opacity-100 transition duration-500"></div>
    </div>
  );
}
