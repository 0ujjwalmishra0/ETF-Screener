from pushbullet import Pushbullet
import subprocess


PUSHBULLET_API_KEY = "o.YtzGePcJGy01z71e8FC3HGaI9nODfGoj"

def send_push_notification(title, message):
    try:
        pb = Pushbullet(PUSHBULLET_API_KEY)
        pb.push_note(title, message)
        print(f"✅ Notification sent: {title}")
    except Exception as e:
        print(f"❌ Notification error: {e}")

def send_mac_notification(title: str, message: str):
    """
    Sends a macOS native notification using AppleScript.
    """
    # Escape double quotes in message for safety
    safe_message = message.replace('"', '\\"')
    subprocess.run([
        "osascript", "-e",
        f'display notification "{safe_message}" with title "{title}"'
    ])



if __name__ == "__main__":
    print("🚀 Running ETF Screener Pro 2026...")
    send_mac_notification("hello","123")