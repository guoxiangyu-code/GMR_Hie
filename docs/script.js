const demoData = {
  null: {
    title: "Locate a red-card moment in this clip.",
    description:
      "The event is semantically plausible in soccer, but the correct answer is an empty set because no red card appears in the clip.",
    gt: [],
    predicted: [],
    emptyNote: "Correct target set: empty"
  },
  single: {
    title: "Find the moment of a corner kick.",
    description:
      "This query corresponds to one temporal segment, matching the classic single-target retrieval setting.",
    gt: [{ left: 34, width: 18 }],
    predicted: [{ left: 35, width: 16 }],
    emptyNote: ""
  },
  multi: {
    title: "Locate all tackle moments by the highlighted team.",
    description:
      "The same semantic event appears multiple times, so the model should return the complete set of relevant moments rather than only one dominant hit.",
    gt: [
      { left: 10, width: 12 },
      { left: 42, width: 16 },
      { left: 73, width: 11 }
    ],
    predicted: [
      { left: 11, width: 10 },
      { left: 43, width: 14 },
      { left: 74, width: 9 }
    ],
    emptyNote: ""
  }
};

const tabs = document.querySelectorAll(".demo-tab");
const titleEl = document.getElementById("demo-title");
const descriptionEl = document.getElementById("demo-description");
const timelineEl = document.getElementById("demo-timeline");

function renderTimeline(items, className) {
  return items
    .map(
      ({ left, width }) =>
        `<span class="demo-segment ${className}" style="left:${left}%; width:${width}%"></span>`
    )
    .join("");
}

function clearEmptyNotes() {
  document.querySelectorAll(".demo-empty-note").forEach((note) => note.remove());
}

function renderDemo(key) {
  const data = demoData[key];
  if (!data || !titleEl || !descriptionEl || !timelineEl) return;

  titleEl.textContent = data.title;
  descriptionEl.textContent = data.description;
  timelineEl.innerHTML = `${renderTimeline(data.gt, "gt")}${renderTimeline(data.predicted, "predicted")}`;

  clearEmptyNotes();

  if (data.emptyNote) {
    timelineEl.insertAdjacentHTML("afterend", `<p class="demo-empty-note">${data.emptyNote}</p>`);
  }
}

function updateTabState(activeKey) {
  tabs.forEach((tab) => {
    const isActive = tab.dataset.demo === activeKey;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });
}

if (tabs.length && titleEl && descriptionEl && timelineEl) {
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const key = tab.dataset.demo;
      updateTabState(key);
      renderDemo(key);
    });
  });

  updateTabState("null");
  renderDemo("null");
}

const revealTargets = document.querySelectorAll(".reveal-on-scroll");

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.14,
      rootMargin: "0px 0px -40px 0px"
    }
  );

  revealTargets.forEach((target) => observer.observe(target));
} else {
  revealTargets.forEach((target) => target.classList.add("is-visible"));
}
