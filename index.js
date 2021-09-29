const {Client, Intents} = require('discord.js');
const { EndBehaviorType, VoiceReceiver, entersState, joinVoiceChannel, VoiceConnection, VoiceConnectionStatus } = require('@discordjs/voice');
const { pipeline } = require('stream');
const { opus }  = require('prism-media');
const { OpusEncoder, OpusDencoder  } = require('@discordjs/opus');
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
                tempMember.send("Ningerian prince needs your help, please save him!")
                const opusStream = receiver.subscribe(tempMember.id, {
                    end: {
                      behavior: EndBehaviorType.AfterSilence,
                      duration: 100,
                    },
                  });
         
                //const rawAudio = opusStream.pipe(new opus.Decoder({ frameSize: 960, channels: 2, rate: 48000 }));
                const oggWriter = new opus.OggLogicalBitstream({
                    opusHead: new opus.OpusHead({
                    channelCount: 2,
                    sampleRate: 48000,
                    }),
                    pageSizeControl: {
                    maxPackets: 10,
                    },
                });
                pipeline(opusStream, oggWriter, createWriteStream('./myfile.ogg'), callback);
            });
        }

	} catch (error) {
		console.warn(error);
	}
}

client.login(token);