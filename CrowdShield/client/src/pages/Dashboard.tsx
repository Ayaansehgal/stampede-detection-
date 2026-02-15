import { useEffect } from "react";

export default function Dashboard() {
  useEffect(() => {
    window.location.href = "http://127.0.0.1:8000/";
  }, []);

  return <div>Redirecting...</div>;
}
