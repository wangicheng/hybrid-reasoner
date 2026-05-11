import { useCallback, useEffect, useRef } from "react";
import { cn } from "../lib/utils";

/**
 * SparklesCore — Canvas-based particle sparkles background.
 * Inspired by Aceternity UI's Sparkles component.
 */
export function SparklesCore({
  id = "sparkles",
  className,
  background = "transparent",
  particleSize = 1.2,
  minSize = 0.6,
  maxSize = 1.4,
  speed = 1,
  particleColor = "#ffffff",
  particleDensity = 120,
}) {
  const canvasRef = useRef(null);
  const particles = useRef([]);
  const animationRef = useRef(null);
  const mousePos = useRef({ x: 0, y: 0 });

  const createParticle = useCallback(
    (width, height) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      size: Math.random() * (maxSize - minSize) + minSize,
      speedX: (Math.random() - 0.5) * speed * 0.3,
      speedY: (Math.random() - 0.5) * speed * 0.3,
      opacity: Math.random() * 0.8 + 0.2,
      opacitySpeed: (Math.random() * 0.01 + 0.003) * speed,
      opacityDir: Math.random() > 0.5 ? 1 : -1,
    }),
    [maxSize, minSize, speed]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const resize = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resize();
    window.addEventListener("resize", resize);

    // Initialize particles
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    particles.current = Array.from({ length: particleDensity }, () =>
      createParticle(w, h)
    );

    const handleMouse = (e) => {
      const rect = canvas.getBoundingClientRect();
      mousePos.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    };
    canvas.addEventListener("mousemove", handleMouse);

    const animate = () => {
      ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);

      particles.current.forEach((p) => {
        // Update
        p.x += p.speedX;
        p.y += p.speedY;
        p.opacity += p.opacitySpeed * p.opacityDir;

        if (p.opacity >= 1 || p.opacity <= 0.1) p.opacityDir *= -1;
        if (p.x < 0) p.x = canvas.offsetWidth;
        if (p.x > canvas.offsetWidth) p.x = 0;
        if (p.y < 0) p.y = canvas.offsetHeight;
        if (p.y > canvas.offsetHeight) p.y = 0;

        // Mouse interaction — particles near mouse glow brighter
        const dx = p.x - mousePos.current.x;
        const dy = p.y - mousePos.current.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const glow = dist < 150 ? (150 - dist) / 150 : 0;

        // Draw
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size + glow * 2, 0, Math.PI * 2);
        ctx.fillStyle = particleColor;
        ctx.globalAlpha = Math.min(1, p.opacity + glow * 0.5);
        ctx.fill();

        if (glow > 0) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size + glow * 6, 0, Math.PI * 2);
          ctx.fillStyle = particleColor;
          ctx.globalAlpha = glow * 0.15;
          ctx.fill();
        }
      });

      ctx.globalAlpha = 1;
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationRef.current);
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("mousemove", handleMouse);
    };
  }, [particleDensity, createParticle, particleColor]);

  return (
    <canvas
      ref={canvasRef}
      id={id}
      className={cn("h-full w-full", className)}
      style={{ background }}
    />
  );
}
