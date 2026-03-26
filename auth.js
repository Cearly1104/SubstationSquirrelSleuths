// Manual account list for local/offline demo access.
// Add or remove users here.
const AUTH_ACCOUNTS = [
  { username: "admin", password: "sss" }
];

const Auth = {
  isAuthenticated() {
    return sessionStorage.getItem("sss_auth") === "1";
  },
  currentUser() {
    return sessionStorage.getItem("sss_user") || "";
  },
  login(username, password) {
    const normalized = (username || "").trim().toLowerCase();
    const match = AUTH_ACCOUNTS.find(
      (acct) => acct.username.toLowerCase() === normalized && acct.password === password
    );
    if (!match) return false;
    sessionStorage.setItem("sss_auth", "1");
    sessionStorage.setItem("sss_user", match.username);
    return true;
  },
  logout() {
    sessionStorage.removeItem("sss_auth");
    sessionStorage.removeItem("sss_user");
    window.location.href = "login.html";
  },
  requireAuth() {
    if (!Auth.isAuthenticated()) {
      const next = encodeURIComponent(window.location.pathname.split("/").pop() + window.location.search);
      window.location.href = `login.html?next=${next}`;
    }
  }
};
