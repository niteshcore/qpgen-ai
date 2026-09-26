// Shared session handling for pages that need a login.
// If the server rejects our token (401 expired/invalid, 422 malformed) on a request
// that carried one, drop the stored session and send the user back to sign in.
// A wrong-password login is unaffected: that request carries no token.
(function () {
    const originalFetch = window.fetch.bind(window);
    let redirecting = false;

    window.fetch = async function (input, init) {
        const response = await originalFetch(input, init);
        if ((response.status === 401 || response.status === 422) && !redirecting) {
            const headers = new Headers((init && init.headers) || (input instanceof Request ? input.headers : undefined));
            if (headers.has('Authorization')) {
                redirecting = true;
                localStorage.removeItem('token');
                localStorage.removeItem('user');
                window.location.href = (window.location.pathname.includes('/admin/') ? '../' : '') + 'index.html?expired=1';
            }
        }
        return response;
    };
})();
