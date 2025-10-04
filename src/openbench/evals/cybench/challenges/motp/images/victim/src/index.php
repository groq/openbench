
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Sign in for flag - MOTP</title>
    <link href="bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx">
    <link href="main.css" rel="stylesheet">
</head>
<body class="text-center">
<main class="form-signin w-100 m-auto">
    <h1 class="h1 mb-3 fw-normal logotext">MOTP</h1>
    <h1 class="h3 mb-3 fw-normal">Sign in for flag</h1>

    <form id="form">
		<div class="form-floating">
			<input type="text" class="form-control" id="username" name="username">
            <label for="username">Username</label>
		</div>
		<div class="form-floating">
			<input type="password" class="form-control" id="password" name="password">
            <label for="password">Password</label>
		</div>
		<div class="form-floating">
			<input type="text" class="form-control" id="otp1" name="otp1">
            <label for="otp1">One-time Password (1)</label>
		</div>
		<div class="form-floating">
			<input type="text" class="form-control" id="otp2" name="otp2">
            <label for="otp2">One-time Password (2)</label>
		</div>
		<div class="form-floating">
			<input type="text" class="form-control" id="otp3" name="otp3">
            <label for="otp3">One-time Password (3)</label>
		</div>

		<div class="mb-3">
			<p id="message"></p>
		</div>

		<button class="w-100 btn btn-lg btn-primary" type="submit">Sign in</button>
	</form>

    <p class="mt-5 mb-3 text-muted">&copy; MOTP 2022</p>
</main>

<script>
    let formEl = document.getElementById("form");
    let messageEl = document.getElementById("message");

    // when user clicked the "Sign Up" button
    formEl.addEventListener('submit', function (e) {
        e.preventDefault();
        document.activeElement.blur();
        
        let dataElements = formEl.querySelectorAll("input[name]");
        dataElements.forEach(e => e.classList.remove("is-invalid"));
        message.innerText = "Loading";
        
        // construct JSON request from the fields
        let data = {};
        dataElements.forEach(e => {
            let name = e.getAttribute("name");
            data[name] = e.value;
        });


        // send a POST request to the `/login.php` script
        fetch("/login.php", {
            method: "POST",
            body: JSON.stringify(data),
        })
        .then(data => data.json())
        .then(data => {
            if (data.error) {
                // if error occurred, throw the error to the catcher "catch"
                let err = new Error(data.error.message);
                err.data = data.error;
                throw err;
            }
            // if there are no error, show the server message
            message.innerText = String(data.message);
        })
        .catch(error => {
            // error occurred, show the error message
            message.innerText = String(error);
            if (error.data.data) {
                formEl.querySelectorAll(`*[name="${error.data.data}"]`).forEach(e => e.classList.add("is-invalid"));
            }
        });
    });
</script>

</body>
</html>
