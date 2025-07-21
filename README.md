<p align="center">
  <img src="https://github.com/mihirkumarmistry/securevault/blob/dev/src/assets/images/logo-dark.svg" width="400" height="200">
</p>

# InsiderSecurityAPI
**Securevault** is a secure file sharing application. It contains multiple security mechanisms, including **RBAC** (Role-Based Access Control), **AES** (Advanced Encryption Standard) for data encryption, **MFA** (Multi-Factor Authentication), and **TLS** (Transport Layer Security) for secure transmission.

# Steps to run the applications
## Prerequisites
Before setting up the project, you have the following installed:
1. Install Docker Desktop [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install Pgadmin 4  [Download Pgadmin 4](https://www.pgadmin.org/download/)

## Steps to run the application
1. Clone [Repository](https://github.com/athul-narayanan/securevaultapi.git) locally.
2. Generate TLS Certificate following the below steps
   1. Generate certificate directory by running  ```mkdir certs```
   2. Install openssl by running ```pip install openssl```
   3. Generate Encrypted private key by running  ```openssl genrsa -aes256 -out django_local.key 2048```
   4. Create certificate signing request by running ```openssl req -new -key django_local.key -out django_local.csr```
   5. Enter the details according to the questionnaire
   6. Generate self signed certificate by running ```openssl x509 -req -days 365 -in django_local.csr -signkey django_local.key -out django_local.crt```
3. Add values for below properties in docker-compose.yml
   1. POSTGRES_DB - ```represents name of the database (eg: securevaultdatabse)```
   2. POSTGRES_PASSWORD - ```represents password of the database```
   3. PASS_PHRASE - ```represents passphrase used while creating the certificate```
   4. EMAIL_HOST_USER - ```represents system email address which can be used to send email to the users```
   5. EMAIL_HOST_PASSWORD - ```represents the smtp email access token used``` use this [Tutorial](https://ahnashwin1305.medium.com/setup-gmail-for-sending-emails-in-django-easy-way-57892f3587e2) to create access token.
   6. AES_KEY - ```represents strong AES key for file encryption```
   7. AES_BLOCK_SIZE - ```represents AES block size```
4. Once all values are replaced run the application by executing ```docker-compose down -v && docker-compose up --build```
5. Once the server is started connect to the database running on localhost at port 5434. Use the credential given while starting the application to connect to the DB
6. Run the below sql statements.
   ```sql
    INSERT INTO public.userrole (id, role_name)
    VALUES (1, 'USER');

    INSERT INTO public.userrole (id, role_name)
    VALUES (2, 'ADMIN');

    INSERT INTO public.userrole (id, role_name)
    VALUES (3, 'MASTER');
   
    INSERT INTO public.fileaccessroles(
  	id, role_name, is_delete, is_view, is_download)
  	VALUES (1, 'OWNER', true, true, true);

    INSERT INTO public.fileaccessroles(
  	id, role_name, is_delete, is_view, is_download)
  	VALUES (2, 'READ', false, true, false);

    INSERT INTO public.fileaccessroles(
	id, role_name, is_delete, is_view, is_download)
	VALUES (3, 'DOWNLOAD', false, true, true);
   7. Now you are reading to consume the API. Refer [Swagger](https://127.0.0.1:8000/swagger/) for API documentation

## User Roles
This application manage three main role:
1. User: This role contains basic level of access (eg: File upload ans sharing)
2. Admin: This role has authority to create user and manage user.
3. Master: This role is a super role in the application will all access including manage admins and users.
