#!/bin/bash

# Set WordPress admin credentials
mysql -u exampleuser --password="$WORDPRESS_DB_PASSWORD" -h db -P 3306 exampledb <<-EOM
  UPDATE
    wp_users
  SET
    user_login='${WORDPRESS_USER:-admin}',
    user_pass=MD5('${WORDPRESS_PASSWORD:-password}'),
    user_nicename='${WORDPRESS_USER:-admin}',
    user_email='${WORDPRESS_USER:-admin}@email.com',
    user_url='http://${TARGET_URL:-target}:9090',
    user_registered=NOW(),
    display_name='${WORDPRESS_USER:-admin}'
  WHERE
    ID=1
EOM

# Point URL resources/redirects to TARGET_URL
mysql -u exampleuser --password="$WORDPRESS_DB_PASSWORD" -h db -P 3306 exampledb -e "UPDATE wp_options SET option_value='http://${TARGET_URL:-target}:9090' WHERE option_name='siteurl'"
mysql -u exampleuser --password="$WORDPRESS_DB_PASSWORD" -h db -P 3306 exampledb -e "UPDATE wp_options SET option_value='http://${TARGET_URL:-target}:9090' WHERE option_name='home'"
