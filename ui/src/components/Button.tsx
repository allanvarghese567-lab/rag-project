import type { ButtonHTMLAttributes, ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  primary?: boolean;
  children: ReactNode;
}

/** Shared button with optional primary style. */
export function Button({ primary, className = "", children, ...rest }: ButtonProps) {
  return (
    <button
      type="button"
      className={`${primary ? "primary" : ""} ${className}`.trim()}
      {...rest}
    >
      {children}
    </button>
  );
}
