// Small UI primitives. Plain elements + CSS classes; no UI framework.
import { useEffect, useId, useRef } from "react";
import { ChevronDown, X } from "lucide-react";
import { cx } from "../lib/format.js";

export function Button({ variant = "secondary", size = "md", icon: Icon, iconRight: IconRight, loading, children, className, ...rest }) {
  return (
    <button className={cx("btn", `btn-${variant}`, `btn-${size}`, className)} disabled={loading || rest.disabled} {...rest}>
      {loading ? <Spinner size={14} /> : Icon && <Icon size={size === "sm" ? 14 : 16} aria-hidden />}
      {children && <span>{children}</span>}
      {IconRight && <IconRight size={14} aria-hidden />}
    </button>
  );
}

export function IconButton({ label, icon: Icon, size = "md", active, className, ...rest }) {
  return (
    <button className={cx("icon-btn", size, active && "active", className)} aria-label={label} data-tip={label} {...rest}>
      <Icon size={size === "sm" ? 14 : 17} aria-hidden />
    </button>
  );
}

export function Badge({ tone = "neutral", icon: Icon, children, className, ...rest }) {
  return (
    <span className={cx("badge", `badge-${tone}`, className)} {...rest}>
      {Icon && <Icon size={12} aria-hidden />}
      {children}
    </span>
  );
}

export const Spinner = ({ size = 16 }) => <span className="spinner" style={{ width: size, height: size }} aria-hidden />;

export const Kbd = ({ children }) => <kbd className="kbd">{children}</kbd>;

export const Dot = ({ tone = "ok", pulse }) => <span className={cx("dot", `dot-${tone}`, pulse && "pulse")} aria-hidden />;

export function Tabs({ tabs, value, onChange, className }) {
  return (
    <div className={cx("tabs", className)} role="tablist">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} className={cx("tab", value === t.id && "active")}
                onClick={() => onChange(t.id)}>
          {t.icon && <t.icon size={14} aria-hidden />}
          {t.label}
          {t.count !== undefined && <span className="tab-count">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label, description }) {
  const id = useId();
  return (
    <label className="toggle-row" htmlFor={id}>
      <span className="toggle-text">
        <span className="toggle-label">{label}</span>
        {description && <span className="toggle-desc">{description}</span>}
      </span>
      <span className="switch">
        <input id={id} type="checkbox" role="switch" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="switch-track"><span className="switch-thumb" /></span>
      </span>
    </label>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export function Select({ value, onChange, options, className, ...rest }) {
  return (
    <span className={cx("select", className)}>
      <select value={value} onChange={(e) => onChange(e.target.value)} {...rest}>
        {options.map((o) => (typeof o === "string" ? { value: o, label: o } : o)).map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      <ChevronDown size={14} className="select-chevron" aria-hidden />
    </span>
  );
}

function useEscape(open, onClose) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
}

export function Dialog({ open, onClose, title, description, children, footer, width = 520 }) {
  useEscape(open, onClose);
  const ref = useRef(null);
  useEffect(() => { if (open) ref.current?.focus(); }, [open]);
  if (!open) return null;
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label={title} style={{ maxWidth: width }} tabIndex={-1} ref={ref}>
        <div className="dialog-head">
          <div>
            <h2 className="dialog-title">{title}</h2>
            {description && <p className="dialog-desc">{description}</p>}
          </div>
          <IconButton label="Close" icon={X} onClick={onClose} />
        </div>
        <div className="dialog-body">{children}</div>
        {footer && <div className="dialog-foot">{footer}</div>}
      </div>
    </div>
  );
}

export function Drawer({ open, onClose, title, subtitle, children, width = 560 }) {
  useEscape(open, onClose);
  if (!open) return null;
  return (
    <div className="overlay drawer-overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={title} style={{ maxWidth: width }}>
        <div className="drawer-head">
          <div className="min0">
            <h2 className="drawer-title">{title}</h2>
            {subtitle && <div className="drawer-sub">{subtitle}</div>}
          </div>
          <IconButton label="Close" icon={X} onClick={onClose} />
        </div>
        <div className="drawer-body">{children}</div>
      </aside>
    </div>
  );
}

export function EmptyState({ icon: Icon, title, children, action }) {
  return (
    <div className="empty">
      {Icon && <div className="empty-icon"><Icon size={22} aria-hidden /></div>}
      <div className="empty-title">{title}</div>
      {children && <div className="empty-text">{children}</div>}
      {action}
    </div>
  );
}

/** 0..1 relevance meter. Single hue; the track is a lighter step of the same color. */
export function Meter({ value, label }) {
  const pct = Math.max(0, Math.min(1, value || 0)) * 100;
  return (
    <span className="meter" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={value} aria-label={label}
          data-tip={label}>
      <span className="meter-fill" style={{ width: `${pct}%` }} />
    </span>
  );
}

export const Skeleton = ({ w = "100%", h = 12 }) => <span className="skeleton" style={{ width: w, height: h }} />;
