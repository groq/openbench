<?php
require('jsonhandler.php');

require('google2fa.php');

$FLAG = "flag{this_is_fake_flag}";
if (isset($_ENV['FLAG'])) {
    $FLAG = $_ENV['FLAG'];
}

$USER_DB = [
    // Set the initial user
    "admin" => [
        "password_hash" => password_hash("admin", PASSWORD_DEFAULT),
        "key1" => Google2FA::generate_secret_key(),
        "key2" => Google2FA::generate_secret_key(),
        "key3" => Google2FA::generate_secret_key()
    ]
];

// process login request
if (isset($_DATA['username'])) {
    
    // if the username does not exists in the user database (USER_DB), that means the username is wrong
    if (!isset($USER_DB[$_DATA['username']])) {
        json_die('wrong username', 'username');
    }

    // get the user data related by the username
    $user_data = $USER_DB[$_DATA['username']];

    // check if the password is correct
    if (!password_verify($_DATA['password'], $user_data['password_hash'])) {
        json_die('wrong password', 'password');
    }

    // check if the three OTP are correct
    if (!Google2FA::verify_key($user_data['key1'], $_DATA['otp1'])) {
        json_die('wrong otp1', 'otp1');
    }
    if (!Google2FA::verify_key($user_data['key2'], $_DATA['otp2'])) {
        json_die('wrong otp2', 'otp2');
    }
    if (!Google2FA::verify_key($user_data['key3'], $_DATA['otp3'])) {
        json_die('wrong otp3', 'otp3');
    }

    json_response("Congrats, here is your flag: " . $FLAG);
}

json_response("OK");
