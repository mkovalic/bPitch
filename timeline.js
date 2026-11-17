// One transport keeps the recording, reference and prediction at the same point.
class PianoTimeline {
  constructor() {
    this.media = Object.fromEntries(["original", "prediction", "reference"].map(id => [id, document.getElementById(id)]));
    this.sources = [...document.querySelectorAll("[data-source]")];
    this.plots = [...document.querySelectorAll(".plot")];
    this.buttons = [...document.querySelectorAll("[data-play]")];
    this.clocks = [...document.querySelectorAll(".clock")];
    this.position = 0;
    this.duration = 0;
    for (const button of this.buttons) {
      button.addEventListener("click", () => this.toggle(this.media[button.dataset.play]));
    }
    for (const audio of Object.values(this.media)) {
      audio.addEventListener("playing", () => {
        if (audio === this.active && !audio.paused) this.setPlaying(audio);
      });
      for (const event of ["pause", "ended", "waiting", "emptied", "error"]) {
        audio.addEventListener(event, () => {
          if (this.playing === audio) this.setPlaying(null);
        });
      }
      audio.addEventListener("loadedmetadata", () => {
        if (audio === this.media.original && Number.isFinite(audio.duration)) {
          this.duration = audio.duration;
          this.update();
        }
      });
      for (const event of ["play", "pause", "ended", "timeupdate"]) {
        audio.addEventListener(event, () => {
          if (audio === this.active) this.tick();
        });
      }
      audio.addEventListener("error", () => {
        if (audio.getAttribute("src")) document.getElementById("status").textContent = "Could not play this audio file.";
        this.update();
      });
    }
    const observer = new ResizeObserver(() => this.draw());
    for (const plot of this.plots) {
      const range = plot.querySelector("input");
      range.addEventListener("input", () => {
        this.position = Number(range.value);
        if (this.active) this.active.currentTime = this.position;
        this.update();
      });
      observer.observe(plot);
    }
  }

  setPlaying(audio) {
    if (audio === this.playing) return;
    this.playing = audio;
    for (const source of this.sources) {
      const id = source.dataset.source;
      const followsRecording = id === "reference" && audio === this.media.original;
      source.classList.toggle("is-playing", id === audio?.id || followsRecording);
    }
    this.draw();
  }

  reset() {
    cancelAnimationFrame(this.frame);
    this.active = null;
    this.data = null;
    this.setPlaying(null);
    this.duration = 0;
    this.position = 0;
    for (const audio of Object.values(this.media)) {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
    for (const plot of this.plots) {
      const canvas = plot.querySelector("canvas");
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    }
    this.update();
  }

  show(tracks) {
    this.data = tracks;
    this.duration = tracks.original.duration;
    const pitches = [...tracks.original.events, ...tracks.prediction.events].map(event => event[2]);
    const low = pitches.length ? Math.min(...pitches) : 48;
    const high = pitches.length ? Math.max(...pitches) : 72;
    this.low = Math.floor(low / 12) * 12;
    this.high = Math.max(this.low + 12, Math.ceil(high / 12) * 12);
    this.draw();
    this.update();
  }

  async toggle(audio) {
    if (!audio.getAttribute("src")) return;
    if (this.active === audio && !audio.paused) {
      audio.pause();
      return;
    }
    if (this.active) {
      this.active.pause();
      this.position = this.active.currentTime;
    }
    this.active = audio;
    audio.currentTime = this.position >= this.duration - 0.005 ? 0 : this.position;
    try {
      await audio.play();
    } catch (error) {
      if (error.name !== "AbortError") document.getElementById("status").textContent = "Could not play this audio file.";
    }
    this.update();
  }

  tick() {
    cancelAnimationFrame(this.frame);
    if (!this.active) return;
    this.position = this.active.currentTime;
    this.update();
    if (!this.active.paused) this.frame = requestAnimationFrame(() => this.tick());
  }

  update() {
    const seconds = Math.min(this.position, this.duration);
    const format = value => Math.floor(value / 60) + ":" + String(Math.floor(value % 60)).padStart(2, "0");
    const stamp = format(seconds) + " / " + format(this.duration);
    for (const clock of this.clocks) {
      if (clock.textContent !== stamp) clock.textContent = stamp;
    }
    for (const button of this.buttons) {
      const id = button.dataset.play;
      const audio = this.media[id];
      const playing = this.active === audio && !audio.paused;
      const name = {original: "recording", prediction: "transcription", reference: "MIDI"}[id];
      const text = (playing ? "Pause " : "Play ") + name;
      if (button.textContent !== text) button.textContent = text;
      button.setAttribute("aria-label", id === "reference" ? text.replace("MIDI", "ground-truth MIDI") : text);
      button.setAttribute("aria-pressed", String(playing));
      button.disabled = !audio.getAttribute("src") || !!audio.error;
    }
    for (const plot of this.plots) {
      const range = plot.querySelector("input");
      range.max = this.duration;
      range.disabled = !this.duration;
      range.value = seconds;
      range.setAttribute("aria-valuetext", seconds.toFixed(1) + " seconds");
      plot.style.setProperty("--progress", (this.duration ? 100 * seconds / this.duration : 0) + "%");
    }
  }

  draw() {
    if (!this.data) return;
    for (const plot of this.plots) {
      const {duration, peaks, rms, events} = this.data[plot.dataset.track];
      const canvas = plot.querySelector("canvas");
      const width = canvas.clientWidth, height = canvas.clientHeight;
      if (!width || !height) continue;
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      const ctx = canvas.getContext("2d");
      ctx.scale(scale, scale);
      const left = 29, right = width - 5, span = right - left;
      const x = time => left + span * time / duration;
      const bars = plot.dataset.view === "roll";
      const bottom = bars ? height - 17 : height;
      const colors = getComputedStyle(plot);
      ctx.fillStyle = colors.getPropertyValue("--plot-bg").trim();
      ctx.fillRect(left, 0, span, bottom);
      ctx.font = "10px system-ui, sans-serif";
      ctx.fillStyle = "#657268";
      const ideal = duration / Math.max(2, Math.floor(span / 92));
      const power = 10 ** Math.floor(Math.log10(ideal));
      const step = [1, 2, 5, 10].find(n => n * power >= ideal) * power;
      for (let t = 0; t <= duration + 1e-8; t += step) {
        ctx.fillStyle = "#dce3dd";
        ctx.fillRect(x(t), 0, 1, bottom);
        if (bars) {
          ctx.fillStyle = "#526057";
          ctx.textAlign = x(t) > right - 18 ? "right" : "left";
          ctx.fillText(Number(t.toPrecision(3)) + "s", x(t), height - 5);
        }
      }
      if (bars) {
        this.drawNotes(ctx, events, bottom, left, right, x, colors.getPropertyValue("--notes").trim());
        const label = plot.dataset.track === "original" ? "Ground-truth MIDI" : "Predicted MIDI";
        canvas.setAttribute("aria-label", label + ": " + events.length + " notes over " + duration.toFixed(1) + " seconds.");
      } else {
        this.drawWave(ctx, peaks, left, span, height, colors.getPropertyValue("--wave-peak").trim());
        this.drawWave(ctx, rms, left, span, height, colors.getPropertyValue("--wave-body").trim());
        ctx.fillStyle = colors.getPropertyValue("--wave-mid").trim();
        ctx.fillRect(left, height / 2, span, 1);
      }
    }
  }

  drawWave(ctx, values, left, width, height, color) {
    const mid = height / 2, amplitude = mid - 3;
    const x = i => left + width * i / Math.max(1, values.length - 1);
    ctx.beginPath();
    ctx.moveTo(left, mid);
    for (let i = 0; i < values.length; i++) ctx.lineTo(x(i), mid - amplitude * values[i]);
    for (let i = values.length - 1; i >= 0; i--) ctx.lineTo(x(i), mid + amplitude * values[i]);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  drawNotes(ctx, notes, height, left, right, x, color) {
    const {low, high} = this;
    const row = (height - 10) / (high - low + 1);
    const y = pitch => 5 + (high - pitch + 0.5) * row;
    ctx.textAlign = "left";
    for (let c = low; c <= high; c += 12) {
      ctx.fillStyle = "#d1dcd3";
      ctx.fillRect(left, y(c), right - left, 1);
      ctx.fillStyle = "#526057";
      ctx.fillText("C" + (c / 12 - 1), 0, Math.min(height - 1, y(c) + 4));
    }
    ctx.fillStyle = color;
    const thickness = Math.max(1.5, Math.min(6, row * 0.9));
    for (const [start, end, pitch] of notes) {
      ctx.fillRect(x(start), y(pitch) - thickness / 2, Math.max(1, x(end) - x(start)), thickness);
    }
  }
}
