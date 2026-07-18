# Frequently Asked Questions

Your destination if you didn't really read the instructions. I get it, I write code I don't even know how to read.

## UPDATE YOUR VERSION

I cannot stress enough this is a **pre-release software**. If I push an update, just download it. Do it, download the update. I have 2 brain cells fighting for 3rd place so I make mistakes that need fixing, and the fixes in this stage are often critical.

It's not hard, I made it check for updates for you so you don't even have to push that little button that already checks for updates for you. It even does the download and reboot automagically. Sure it takes a minute, but it either takes a minute and you can use it, or you skip the update and it doesn't fucking work. So please, update.

## I just updated and it still doesn't work??

Sometimes my fuck ups are in the wiki scraper tool, or in the way wiki data is stored. If that happens, you will need to click the "Update icons" button in the bottom left. If it's still not working, you may have a problem.

## "Does I have a problem?"

You play DBD enough to download and read the FAQ of a bloodpoint spender. Yes you have a problem. But let's make sure your software doesn't. Speed run things to check that take literally seconds:

- Ensure "Use simulator" and "Dry run" in the run tab are **unchecked**. These modes don't actually click anything. They're a safety net for my legal liability to you (probably).
- Make sure the bloodweb is visible on screen with nothing blocking it. dbdbp-pas only sees what you can see, so watching a youtube video while this runs would need a second monitor.
- Check your game is in either windowed mode, or windowed full screen. Normal full screen causes issues with reading kernel level input as this software does.
- If the first live run is slow, wait. It is building a cache it can reference from for matching efficiency, and will only need to do this slow thing after first downloading and any updates.
- If you changed the Start/Pause and Stop keybinds but they're not working, you need to restart the software for the keybinds to take effect. Don't ask me why I just work here.
- Double check your priority profile looks right, **and make sure to save it**.


### HEY YOU MISSED STILL THE BLOODWEB NODE I WANT

First of all, take some xanax and stop yelling at me I did this for free asshole. There's a few things that could cause this, First check the common problem solutions above. If those didn't help, you're either a lost cause, or I must congratulate you on your discovery of my failures. See the bottom of the Instructions tab in the UI to see how to use the debugger and submit an issue on github to put my ass to work.