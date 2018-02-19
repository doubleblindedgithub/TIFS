import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

def send_mail(user, pwd, to, smtp='smtp.gmail.com', subject='Status update', txt=''):

    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = to
    msg['Subject'] = subject
    msg.attach(MIMEText(txt))

    server = smtplib.SMTP(smtp, 587)
    server.ehlo()
    server.starttls()
    server.login(user, pwd)
    server.sendmail(user, to, msg.as_string())
    server.close()
