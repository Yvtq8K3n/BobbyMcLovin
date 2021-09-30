const fs = require ('fs');
const {Client, Intents} = require('discord.js');
const { EndBehaviorType, VoiceReceiver, entersState, joinVoiceChannel, VoiceConnection, VoiceConnectionStatus } = require('@discordjs/voice');
const { OpusEncoder, OpusDencoder} = require('@discordjs/opus');
const { opus }  = require('prism-media');
const { Transform } = require('stream');
const { FileWriter } = require('wav')
const {token} = require('./auth.json');

const client = new Client({ intents: [Intents.FLAGS.GUILDS, Intents.FLAGS.GUILD_MESSAGES, Intents.FLAGS.GUILD_VOICE_STATES] });

client.on('ready', () => {
    console.log(`Logged in as ${client.user.tag}`);
})

//message no longer works so use messageCreate
client.on('messageCreate', msg => {
    if (msg.content.startsWith('!on')) {
        start(msg);
    }
})


async function start(msg){
    let member = msg.member
    const voiceChannel = member.voice.channel
    connection = joinVoiceChannel({
        channelId: voiceChannel.id,
        guildId: voiceChannel.guild.id,
        selfDeaf: false,
        selfMute: false,
        adapterCreator: voiceChannel.guild.voiceAdapterCreator,
    });
    try{
        await entersState(connection, VoiceConnectionStatus.Ready, 20e3);
		if (connection) {	    
            const receiver = connection.receiver;
            member.guild.channels.cache.get(voiceChannel.id).members.forEach((tempMember) => {
                const username = findUsername(tempMember.id)
                const filename = `./recordings/${Date.now()}_${username}.wav`;

                const encoder = new OpusEncoder(16000, 1)
                const commandAudioStream = receiver.subscribe(tempMember.id, {
                    end: {
                      behavior: EndBehaviorType.AfterSilence,
                      duration: 100,
                    },
                })
                .pipe(new OpusDecodingStream({}, encoder))
                .pipe(new FileWriter(filename, {
                    channels: 1,
                    sampleRate: 16000
                }))
                //const rawAudio = opusStream.pipe(new opus.Decoder({ frameSize: 960, channels: 2, rate: 48000 }));
                /*
                const oggStream = new opus.OggLogicalBitstream({
                    opusHead: new opus.OpusHead({
                        channelCount: 2,
                        sampleRate: 48000,
                    }),
                    opusTags: new opus.OpusTags({
                        maxPackets: 10,
                    })
                });
                
                
                const out = fs.createWriteStream(filename);

                pipeline(opusStream, oggStream, out,  (err) => {
                    console.log(err);
                    if (err) {
                        console.warn('❌ Error recording file ${filename} - ${err.message}');
                    } else {
                        console.log('✅ Recorded ${filename}');
                    }
                });*/
            });
        }

	} catch (error) {
		console.warn(error);
	}
}

function findUsername(userId){
    const User = client.users.cache.get(userId);
    if (User) { // Checking if the user exists.
       return User.tag;
    } else {
        message.channel.send("User not found.") // The user doesn't exists or the bot couldn't find him.
    };
}

class OpusDecodingStream extends Transform {
    encoder

    constructor(options, encoder) {
        super(options)
        this.encoder = encoder
    }

    _transform(data, encoding, callback) {
        this.push(this.encoder.decode(data))
        callback()
    }
}

client.login(token);