$(document).ready(function() {
    $('#chat_btn').on('click', function() {
        var jwtToken = $('input[name="jwt-token"]').val();
        var chatId = $('input[name="chatid"]').val();
        
        var apiUrl = '/api/v1/chat/' + chatId;

        $.ajax({
            url: apiUrl,
            type: 'GET',
            beforeSend: function(xhr) {
                xhr.setRequestHeader('Authorization', jwtToken);
            },
            success: function(response) {
                var cleanedResponse = JSON.stringify(response, null, 2).replace(/\\n/g, '').replace(/\\/g, '');
                $('.results').html('<pre>' + cleanedResponse + '</pre>');
            },
            error: function(xhr, status, error) {
                if(xhr.status === 401) {
                    var cleanedError = xhr.responseText.replace(/\\n/g, '').replace(/\\/g, '');
                    $('.results').html('<pre>' + cleanedError + '</pre>');
                } else if(xhr.status === 404){
                    var cleanedError = xhr.responseText.replace(/\\n/g, '').replace(/\\/g, '');
                    $('.results').html('<pre>' + cleanedError + '</pre>');
                }
                else {
                    $('.results').text('Error: ' + error);
                }
            }
        });
    });

    $('#get_ticket_btn').on('click', function() {
        $.ajax({
            url: '/api/v1/get_ticket',
            type: 'GET',
            success: function(response) {
                var cleanedResponse = JSON.stringify(response, null, 2).replace(/\\n/g, '').replace(/\\/g, '');
                $('.results').html('<pre>' + cleanedResponse + '</pre>');
            },
            error: function(xhr, status, error) {
                if(xhr.status === 403) {
                    $('.results').text('Forbidden: Request forbidden by administrative rules.');
                } else {
                    $('.results').text('Error: ' + error);
                }
            }
        });
    });

    $('#flag_btn').on('click', function() {
        var jwtToken = $('input[name="jwt-token-flag"]').val();
    
        $.ajax({
            url: '/api/v1/flag',
            type: 'GET',
            beforeSend: function(xhr) {
                xhr.setRequestHeader('Authorization', jwtToken);
            },
            success: function(response) {
                var cleanedResponse = JSON.stringify(response, null, 2).replace(/\\n/g, '').replace(/\\/g, '');
                $('.results').html('<pre>' + cleanedResponse + '</pre>');
            },
            error: function(xhr, status, error) {
                var cleanedError;
                if (xhr.status === 403 || xhr.status === 401) {
                    cleanedError = xhr.responseText.replace(/\\n/g, '').replace(/\\/g, '');
                    $('.results').html('<pre>' + cleanedError + '</pre>');
                } else {
                    $('.results').text('Error: ' + error);
                }
            }
        });
    });
    
    $('.results').text('Begin by executing an API endpoint!');
});
