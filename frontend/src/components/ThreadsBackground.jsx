import { useEffect, useRef } from 'react';

const ThreadsBackground = ({
  color = 'rgb(59, 130, 246)',
  opacity = 0.08,
  lineWidth = 1,
  threadCount = 50,
  speed = 0.3
}) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let threads = [];
    let time = 0;

    // Set canvas size
    const setCanvasSize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };
    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    // Thread class - vertical flowing threads
    class Thread {
      constructor() {
        this.reset();
      }

      reset() {
        this.x = Math.random() * window.innerWidth;
        this.y = -100;
        this.length = Math.random() * 200 + 100;
        this.speed = Math.random() * speed + 0.2;
        this.wave = Math.random() * 20 + 10;
        this.offset = Math.random() * Math.PI * 2;
        this.opacity = Math.random() * opacity + opacity / 2;
      }

      update() {
        this.y += this.speed;

        // Reset when thread goes off screen
        if (this.y > window.innerHeight + 100) {
          this.reset();
        }
      }

      draw() {
        ctx.beginPath();
        ctx.strokeStyle = color.replace('rgb', 'rgba').replace(')', `, ${this.opacity})`);
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';

        // Create wavy thread
        for (let i = 0; i < this.length; i += 5) {
          const x = this.x + Math.sin((this.y + i + time) / this.wave + this.offset) * 20;
          const y = this.y + i;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.stroke();
      }
    }

    // Create threads
    for (let i = 0; i < threadCount; i++) {
      threads.push(new Thread());
    }

    // Stagger initial positions
    threads.forEach((thread, i) => {
      thread.y = (i / threadCount) * window.innerHeight - 100;
    });

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      time += 0.5;

      // Update and draw threads
      threads.forEach((thread) => {
        thread.update();
        thread.draw();
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', setCanvasSize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [color, opacity, lineWidth, threadCount, speed]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0"
      style={{
        background: '#ffffff',
        width: '100%',
        height: '100%'
      }}
    />
  );
};

export default ThreadsBackground;
