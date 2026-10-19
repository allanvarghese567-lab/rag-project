interface ToastProps {
  message: string;
  error?: boolean;
  visible: boolean;
}

/** Fixed bottom toast notification. */
export function Toast({ message, error, visible }: ToastProps) {
  return (
    <div
      className={`toast${visible ? " show" : ""}${error ? " error" : ""}`}
      role="status"
    >
      {message}
    </div>
  );
}
