import { useCallback, useEffect, useRef, useState } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/gif"];

export default function UploadPanel({ onFileSelected, disabled }) {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  // Revoke the previous object URL whenever it's replaced or the panel unmounts.
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const handleFile = useCallback(
    (file) => {
      if (!file) return;
      if (!ACCEPTED_TYPES.includes(file.type)) {
        onFileSelected(null, "Please choose a JPEG, PNG, BMP, or GIF image.");
        return;
      }
      setPreviewUrl(URL.createObjectURL(file));
      onFileSelected(file, null);
    },
    [onFileSelected]
  );

  return (
    <div
      className={`upload-panel ${isDragging ? "dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFile(e.dataTransfer.files?.[0]);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        hidden
        disabled={disabled}
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      {previewUrl ? (
        <div className="preview-wrapper">
          <img src={previewUrl} alt="Selected panel preview" className="preview-image" />
          <p className="upload-subtext">Click to choose a different photo</p>
        </div>
      ) : (
        <div className="upload-placeholder">
          <p>Drag & drop a solar panel photo here</p>
          <p className="upload-subtext">or click to browse</p>
        </div>
      )}
    </div>
  );
}
