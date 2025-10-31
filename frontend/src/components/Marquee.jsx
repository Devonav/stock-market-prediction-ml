import { useEffect, useRef } from 'react';
import './Marquee.css';

const Marquee = ({
  children,
  speed = 50,
  direction = 'left',
  pauseOnHover = false,
  className = ''
}) => {
  const marqueeRef = useRef(null);

  useEffect(() => {
    const marquee = marqueeRef.current;
    if (!marquee) return;

    const content = marquee.querySelector('.marquee-content');
    const clone = content.cloneNode(true);
    marquee.appendChild(clone);

    return () => {
      if (marquee.contains(clone)) {
        marquee.removeChild(clone);
      }
    };
  }, [children]);

  return (
    <div
      className={`marquee ${className} ${pauseOnHover ? 'pause-on-hover' : ''}`}
      ref={marqueeRef}
      style={{
        '--marquee-speed': `${speed}s`,
        '--marquee-direction': direction === 'right' ? 'reverse' : 'normal'
      }}
    >
      <div className="marquee-content">
        {children}
      </div>
    </div>
  );
};

export default Marquee;
