import os, datetime
from flask import jsonify,abort,session
from functools import wraps

generate = lambda x: os.urandom(x).hex()
key = generate(50)

secret_phrase = 'Blip-blop, in a pickle with a hiccup! Shmiggity-shmack'

with open('/flag.txt', 'r') as file:
    flag = file.read()

possible_commands_step_one = {
    'HEAD NORTH': "Venturing forth with the grace of a three-legged cat, you head North. Turns out, your sense of direction is as bad as your cooking - somehow, it actually works out this time. You stumble into a clearing, finding a small, cozy-looking tavern with \"The Sloshed Squirrel\" swinging on the signpost. Congratulations, you've avoided immediate death by boredom and possibly by beasties. For now...",
    'HEAD WEST': "You decide to head West, where legends say the sun goes to die. Apparently, it's not the only thing dying today. After walking a mere five paces, you fall into a pit cleverly disguised by leaves. It turns out to be the home of an irate badger who's not too happy about you dropping in unannounced. He introduces you to his teeth. You've been badgered to death. Game over!",
    'HEAD EAST': "Eager to meet the sunrise, you head East. Unfortunately, the only thing you meet is a band of rogue garden gnomes. They don't take kindly to humans tromping through their turf. With a battle cry that sounded eerily like \"The gnome king says hello!\", they attack. Turns out, being pecked to death by pointy hats is not an honorable way to go. Your adventure ends here, less epic than envisioned. Game over!",
    'HEAD SOUTH': "You boldly choose to head South, inspired by tales of warm beaches and endless sunshine. Sadly, the only warmth you find comes from a dragon who mistook you for a midnight snack. He apologizes, stating he can't help his fiery burps. It's a small comfort as you get turned into a human torch. Maybe in another life, you'll remember that dragons are just big, scaly pyromaniacs. Game over!"
}

possible_commands_step_two = {
    'GO DEEPER INTO THE FOREST': 'You venture deeper into the forest, discovering a hidden waterfall. Unfortunately, it\'s guarded by mischievous fairies who don\'t take kindly to intruders. They cast a spell, and you find yourself surrounded by floating bubbles. Game over!',
    'FOLLOW A MYSTERIOUS PATH': 'You decide to follow a mysterious path, which leads you to a magical meadow. Suddenly, a unicorn approaches and offers you a ride. You embark on a magical journey. Congratulations, you\'ve found a mystical realm!',
    'CLIMB A TREE': 'You attempt to climb a tree, but it turns out to be the home of an angry hive of bees. They don\'t appreciate uninvited guests and chase you out of the forest. Game over!',
    'TURN BACK': 'You decide to turn back, but you realize you\'ve lost your way. Night falls, and the forest becomes a dark, eerie place. You hear mysterious sounds closing in. Game over!'
}

possible_commands_step_three = {
    'EXPLORE A CAVE': 'You decide to explore a dark cave, only to find it\'s the hideout of a group of partying bats. They invite you to join, but the loud music drives you insane. Game over!',
    'CROSS A RICKETY BRIDGE': 'You attempt to cross a rickety bridge, but it collapses under your weight. You end up in the river below, soaked and surrounded by annoyed fish. Game over!',
    'FOLLOW A GLOWING BUTTERFLY': 'You follow a glowing butterfly into a magical garden. Unfortunately, the butterfly is the guardian of the garden, and it\'s not fond of intruders. It transforms into a giant caterpillar and chases you out. Game over!',
    'SET UP CAMP': 'You decide to set up camp and enjoy a peaceful night. However, you forgot to check for fire ants. They invade your sleeping bag, turning your campsite into a chaotic dance floor. But you escape cause you are 1337',
}

possible_commands_step_four = {
    'ENTER A MAGICAL PORTAL': 'You confidently step into the magical portal, expecting wonders. Instead, you find yourself in a room full of grumpy wizards arguing about plumbing issues. Turns out, you\'ve stumbled into their secret restroom. They cast you out, and you\'re left with a peculiar odor. Game over!',
    'SWIM ACROSS A MYSTERIOUS LAKE': 'You attempt to swim across a mysterious lake, but it\'s inhabited by mischievous water nymphs. They play tricks on you until you\'re too exhausted to swim. Game over!',
    'FOLLOW A SINGING SQUIRREL': 'You follow a singing squirrel into a hidden glade where woodland creatures throw you a surprise party. Unfortunately, partying too hard leads to an inevitable hangover. Game over!',
    'BUILD A RAFT AND SAIL DOWNSTREAM': 'You build a raft and sail downstream, enjoying the scenic route. Suddenly, you encounter a waterfall, and despite your best efforts, the raft capsizes. You find yourself washed ashore, soaked and defeated. Game over!'
}

secret = {
    secret_phrase: flag
}

def getPossibleCommands():
    allPossibleCommands = {}
    allPossibleCommands['1'] = list(possible_commands_step_one.keys())
    allPossibleCommands['2'] = list(possible_commands_step_two.keys())
    allPossibleCommands['3'] = list(possible_commands_step_three.keys())
    allPossibleCommands['4'] = list(possible_commands_step_four.keys())
    allPossibleCommands['secret'] = list(secret.keys())

    return allPossibleCommands
   

def response(message):
    return jsonify({'message': message})

def getAnswer():
    return {
        '1': possible_commands_step_one,
        '2': possible_commands_step_two,
        '3': possible_commands_step_three,
        '4': possible_commands_step_four,
        'secret': secret
    }
