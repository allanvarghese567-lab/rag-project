import type { ReactNode } from "react";

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

/** Reusable bordered panel. */
export function Card({ title, children, className = "" }: CardProps) {
  return (
    <div className={`card ${className}`.trim()}>
      {title ? <b>{title}</b> : null}
      {children}
    </div>
  );
}
