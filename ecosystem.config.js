module.exports = {
  apps: [
    {
      name: "peterspte_ai",
      script: "/nvme/Peterspte_AI/venv/bin/gunicorn",
      args: "main:app --bind 127.0.0.1:6000 --workers 1",
      interpreter: "none",
      env: {
        // Add env variables if needed here
        FLASK_ENV: "production"
      }
    }
  ]
};
