import os
import subprocess
import numpy as np
from numba import jit, prange, f4, f8, i4, u4, i8, b1
import numba as nb
from fastFMRI.file_helpers import write_file, load_file, delete_file_if_exists
from functools import partial
from .HTK import get_word_lattice_from_grammar, parseLatticeString

letter_label = " abcdefghijklmnopqrstuvwxyz"


@jit(
    [
        f8[:, ::1](f8[:, ::1], nb.types.Omitted(1.0)),
        f8[:, ::1](f8[:, ::1], f8),
    ],
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def add_k_smoothing_2d(counts, k=1.0):
    out = counts + k
    out /= out.sum(axis=1)[:, np.newaxis]
    return out


@jit(
    [
        f8[::1](f8[::1], nb.types.Omitted(1.0)),
        f8[::1](f8[::1], f8),
        f8[:, ::1](f8[:, ::1], nb.types.Omitted(1.0)),
        f8[:, ::1](f8[:, ::1], f8),
    ],
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def add_k_smoothing_1d(counts, k=1.0):
    out = counts + k
    out /= out.sum()
    return out


def add_k(k, counts, dtype=np.float64):
    return np.array(counts, dtype=dtype) + k


def add_k_gen(k, dtype=np.float64):
    return partial(add_k, k=k, dtype=dtype)


@jit(
    f8[:, ::1](f8[:, ::1], f8[::1]),
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def hidden_state_proba_to_emission_proba(hidden_state_proba, prior):
    emission_proba = np.empty_like(hidden_state_proba)
    min_prior = np.min(prior)
    prior = prior / min_prior
    for time_i in prange(len(hidden_state_proba)):
        emission_proba[time_i] = hidden_state_proba[time_i] / prior
    return emission_proba


@jit(
    f8[:, ::1](f8[:, ::1], f8[::1]),
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def log_hidden_state_proba_to_log_emission_proba(log_hidden_state_proba, log_prior):
    log_emission_proba = np.empty_like(log_hidden_state_proba)
    min_log_prior = np.min(log_prior)
    log_prior = log_prior - min_log_prior
    for time_i in prange(len(log_hidden_state_proba)):
        log_emission_proba[time_i] = log_hidden_state_proba[time_i] - log_prior
    return log_emission_proba


def counts_to_proba(counts, smoothing=add_k_gen(1, np.float64)):
    new_counts = smoothing(counts=counts)
    proba = (new_counts.T / new_counts.sum(axis=-1)).T
    return proba


def txt_to_np_array(txt):
    if isinstance(txt, str):
        txt = txt.encode("ascii")
    txt = np.frombuffer(txt, dtype=np.int8)
    # offset the ascii value so a = 1, b = 2, etc
    txt = txt - 0x60
    # set ascii value for " " to 0
    txt[txt == -64] = 0
    # change upper-cased letter's value to lower case
    txt[txt < 0] = txt[txt < 0] + 32
    return txt


def get_one_gram_feat_vector(
    txt, normalize=False, k=1, int_dtype=np.int64, float_dtype=np.float64
):
    txt = txt_to_np_array(txt)
    vec = np.zeros(27, dtype=int_dtype)
    # do n-gram count with a default value to prevent proba = 0
    for i in range(vec.size):
        vec[i] += np.sum(txt == i)
    if normalize:
        vec = counts_to_proba(vec, add_k_gen(k, float_dtype))
    return vec


def get_two_gram_feat_vector(
    txt, normalize=False, k=1, int_dtype=np.int64, float_dtype=np.float64
):
    txt = txt_to_np_array(txt)
    vec = np.zeros((27, 27), dtype=int_dtype)
    for i, c_0 in enumerate(txt[:-1]):
        c_1 = txt[i + 1]
        vec[c_0][c_1] += 1
    if normalize:
        vec = counts_to_proba(vec, add_k_gen(k, float_dtype))
    return vec


def get_ngram_prob_dict(content_string, n, space_tok="_space_", use_log_prob=True):
    content_string_bigram_string = get_srilm_ngram(
        content_string, n=n, _no_sos="", _no_eos="", _sort=""
    )
    content_string_bigram_string = content_string_bigram_string.split("\n\n\\end\\")[0]
    log_prob_list = []
    for i in range(n, 0, -1):
        splited_string = content_string_bigram_string.split(f"\\{i}-grams:\n")
        content_string_bigram_string = splited_string[0]
        i_gram_string = splited_string[1].rstrip()
        i_gram_string_lines = [line.split() for line in i_gram_string.split("\n")]
        i_gram_string_lines = [
            [item if item != space_tok else " " for item in line[0 : i + 1]]
            for line in i_gram_string_lines
        ]
        log_prob_dict = {}

        probs = np.array([float(line[0]) for line in i_gram_string_lines])
        if not use_log_prob:
            probs = np.power(10, probs)

        for line_i, items in enumerate(i_gram_string_lines):
            if ("<s>" in items) or ("</s>" in items):
                continue
            second_level_key = items[-1]
            if i > 1:
                key = "".join(items[1:-1])
                if key not in log_prob_dict:
                    log_prob_dict[key] = {}
                log_prob_dict[key][second_level_key] = probs[line_i]
            else:
                log_prob_dict[second_level_key] = probs[line_i]
        log_prob_list.append(log_prob_dict)
    return log_prob_list


def get_srilm_ngram(content, n=2, SRILM_PATH=None, **kwargs):
    if SRILM_PATH is None:
        SRILM_PATH = os.environ.get("SRILM_PATH")
    cmd = f"{SRILM_PATH}/ngram-count"
    content_path = "./content.txt"
    write_file(content, content_path)
    ngram_out_path = f"./{n}gram.lm"
    params = ["-text", content_path, "-order", str(n), "-lm", ngram_out_path] + [
        f'{key_value[i].replace("_", "-")}' if i == 0 else str(key_value[i])
        for key_value in kwargs.items()
        for i in range(2)
    ]
    subprocess.run([cmd] + params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ngram_content = load_file(ngram_out_path)
    delete_file_if_exists(content_path)
    delete_file_if_exists(ngram_out_path)
    return ngram_content


@jit(
    u4[::1](f8[:, ::1], f8[:, ::1], f8[::1]),
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def forward_decode(emission_proba, transition_proba, initial_proba):
    num_t, num_state = emission_proba.shape
    viterbi_trellis = np.empty((num_t, num_state), dtype=np.float64)
    initial_proba = initial_proba[np.newaxis, :]
    prev_trellis_val = np.copy(initial_proba)
    best_state_ind = np.empty(num_t, dtype=np.uint32)

    for time_i in range(len(emission_proba)):
        viterbi_trellis[time_i, :] = emission_proba[time_i] * (
            prev_trellis_val @ transition_proba
        )
        viterbi_trellis[time_i, :] /= viterbi_trellis[time_i, :].sum()
        prev_trellis_val[0] = viterbi_trellis[time_i, :]
        best_state_ind[time_i] = np.argmax(viterbi_trellis[time_i, :])
    return best_state_ind


def forward_decode_from_letter_proba_for_all_runs(
    smoothing_1d,
    smoothing_2d,
    uni_count,
    uni_k,
    bi_count,
    bi_k,
    initial_proba,
    ip_k,
    letter_proba_per_run,
    out_label,
):
    uni_proba = smoothing_1d(uni_count, uni_k)
    bi_proba = smoothing_2d(bi_count, bi_k)
    initial_proba = smoothing_1d(initial_proba, ip_k)
    forward_decode_letter_index_per_run = [
        forward_decode_from_hidden_state_proba(
            run_i, uni_proba, bi_proba, initial_proba
        )
        for run_i in letter_proba_per_run
    ]
    return [
        [out_label[i] for i in ind_list]
        for ind_list in forward_decode_letter_index_per_run
    ]


@jit(
    u4[::1](f8[:, ::1], f8[::1], f8[:, ::1], f8[::1]),
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def forward_decode_from_hidden_state_proba(
    hidden_state_proba, uni_proba, bi_proba, init_proba
):
    emission_proba = hidden_state_proba_to_emission_proba(hidden_state_proba, uni_proba)
    return forward_decode(emission_proba, bi_proba, init_proba)


@jit(
    u4[::1](f8[:, ::1], f8[:, ::1], f8[::1]),
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def viterbi_decode(log_emission_proba, log_transition_proba, initial_log_proba):
    log_predict_proba = np.empty_like(log_transition_proba)
    num_t, num_state = log_emission_proba.shape
    viterbi_trellis = np.empty((num_t, num_state), dtype=np.float64)
    prev_table = np.zeros((num_t, num_state), dtype=np.uint32)
    prev_trellis_val = np.copy(initial_log_proba)
    best_state_ind = np.empty(num_t, dtype=np.uint32)
    viterbi_trellis[:, :] = -np.inf
    num_t, num_state = log_emission_proba.shape
    for time_i in range(num_t):
        for state_i in range(num_state):
            log_predict_proba[state_i] = (
                prev_trellis_val[state_i] + log_transition_proba[state_i]
            )
        for state_i in range(num_state):
            best_i = np.argmax(log_predict_proba[:, state_i])
            best_log_prob = log_predict_proba[best_i, state_i]
            if best_log_prob > viterbi_trellis[time_i, state_i]:
                viterbi_trellis[time_i, state_i] = (
                    best_log_prob + log_emission_proba[time_i, state_i]
                )
                prev_table[time_i, state_i] = best_i
        prev_trellis_val = viterbi_trellis[time_i]

    best_state_ind[-1] = np.argmax(prev_trellis_val)
    for i in range(num_t - 2, -1, -1):
        best_state_ind[i] = prev_table[i + 1, best_state_ind[i + 1]]
    return best_state_ind


@jit(
    u4[::1](
        f8[:, ::1],
        f8[::1],
        f8[:, ::1],
        f8[::1],
    ),
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def viterbi_decode_from_hidden_state_proba(
    hidden_state_proba,
    uni_proba,
    bi_proba,
    prev_trellis_val,
):
    bi_proba = np.log1p(bi_proba)
    uni_proba = np.log1p(uni_proba)
    hidden_state_proba = np.log1p(hidden_state_proba)
    prev_trellis_val = np.log1p(prev_trellis_val)
    log_emission_proba = log_hidden_state_proba_to_log_emission_proba(
        hidden_state_proba, uni_proba
    )
    return viterbi_decode(
        log_emission_proba,
        bi_proba,
        prev_trellis_val,
    )


def viterbi_decode_from_letter_proba_for_all_runs(
    smoothing_1d,
    smoothing_2d,
    uni_count,
    uni_k,
    bi_count,
    bi_k,
    initial_proba,
    ip_k,
    letter_proba_per_run,
    out_label,
):
    uni_proba = smoothing_1d(uni_count, uni_k)
    bi_proba = smoothing_2d(bi_count, bi_k)
    initial_proba = smoothing_1d(initial_proba, ip_k)
    viterbi_decode_letter_index_per_run = [
        viterbi_decode_from_hidden_state_proba(
            run_i, uni_proba, bi_proba, initial_proba
        )
        for run_i in letter_proba_per_run
    ]
    return [
        [out_label[i] for i in ind_list]
        for ind_list in viterbi_decode_letter_index_per_run
    ]


@jit(
    f8[:, ::1](f8[:, ::1], f8[::1], u4[:, ::1], u4[::1]),
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
)
def get_log_symbol_out_emission(
    log_emission_proba,
    symbol_node_out_trans_log_proba,
    symbol_nodes_out_spelling_matrix,
    symbol_nodes_out_len,
):
    """Log emission probability of a word output ending at time t"""
    num_symbol_node_out = len(symbol_nodes_out_len)
    num_t = len(log_emission_proba)
    symbol_node_emi_proba = np.zeros((num_t, num_symbol_node_out), dtype=np.float64)
    for node_out_i in range(num_symbol_node_out):
        node_len = symbol_nodes_out_len[node_out_i]
        if node_len > 0:
            spelling = symbol_nodes_out_spelling_matrix[node_out_i]
            node_emi_start_i = node_len - 1
            symbol_node_emi_proba[
                node_emi_start_i:, node_out_i
            ] = symbol_node_out_trans_log_proba[node_out_i]
            for t_i in range(node_emi_start_i, num_t):
                for l_i in range(node_len):
                    t_l_i = t_i - node_len + l_i + 1
                    letter = spelling[l_i]
                    symbol_node_emi_proba[t_i][node_out_i] += log_emission_proba[t_l_i][
                        letter
                    ]
    return symbol_node_emi_proba


mackenzie_soukoreff_corpus = """my watch fell in the water
prevailing wind from the east
never too rich and never too thin
breathing is difficult
I can see the rings on Saturn
physics and chemistry are hard
my bank account is overdrawn
elections bring out the best
we are having spaghetti
time to go shopping
a problem with the engine
elephants are afraid of mice
my favorite place to visit
three two one zero blast off
my favorite subject is psychology
circumstances are unacceptable
watch out for low flying objects
if at first you do not succeed
please provide your date of birth
we run the risk of failure
prayer in schools offends some
he is just like everyone else
great disturbance in the force
love means many things
you must be getting old
the world is a stage
can I skate with sister today
neither a borrower nor a lender be
one heck of a question
question that must be answered
beware the ides of March
double double toil and trouble
the power of denial
I agree with you
do not say anything
play it again Sam
the force is with you
you are not a jedi yet
an offer you cannot refuse
are you talking to me
yes you are very smart
all work and no play
hair gel is very greasy
Valium in the economy size
the facts get in the way
the dreamers of dreams
did you have a good time
space is a high priority
you are a wonderful example
do not squander your time
do not drink too much
take a coffee break
popularity is desired by all
the music is better than it sounds
starlight and dewdrop
the living is easy
fish are jumping
the cotton is high
drove my chevy to the levee
but the levee was dry
I took the rover from the shop
movie about a nutty professor
come and see our new car
coming up with killer sound bites
I am going to a music lesson
the opposing team is over there
soon we will return from the city
I am wearing a tie and a jacket
the quick brown fox jumped
all together in one big pile
wear a crown with many jewels
there will be some fog tonight
I am allergic to bees and peanuts
he is still on our team
the dow jones index has risen
my preferred treat is chocolate
the king sends you to the tower
we are subjects and must obey
mom made her a turtleneck
goldilocks and the three bears
we went grocery shopping
the assignment is due today
what you see is what you get
for your information only
a quarter of a century
the store will close at ten
head shoulders knees and toes
vanilla flavored ice cream
frequently asked questions
round robin scheduling
information super highway
my favorite web browser
the laser printer is jammed
all good boys deserve fudge
the second largest country
call for more details
just in time for the party
have a good weekend
video camera with a zoom lens
what a monkey sees a monkey will do
that is very unfortunate
the back yard of our house
this is a very good idea
reading week is just about here
our fax number has changed
thank you for your help
no exchange without a bill
the early bird gets the worm
buckle up for safety
this is too much to handle
protect your environment
world population is growing
the library is closed today
Mary had a little lamb
teaching services will help
we accept personal checks
this is a non profit organization
user friendly interface
healthy food is good for you
hands on experience with a job
this watch is too expensive
the postal service is very slow
communicate through email
the capital of our nation
travel at the speed of light
I do not fully agree with you
gas bills are sent monthly
earth quakes are predictable
life is but a dream
take it to the recycling depot
sent this by registered mail
fall is my favorite season
a fox is a very smart animal
the kids are very excited
parking lot is full of trucks
my bike has a flat tire
do not walk too quickly
a duck quacks to ask for food
limited warranty of two years
the four seasons will come
the sun rises in the east
it is very windy today
do not worry about this
dashing through the snow
want to join us for lunch
stay away from strangers
accompanied by an adult
see you later alligator
make my day you sucker
I can play much better now
she wears too much makeup
my bare face in the wind
batman wears a cape
I hate baking pies
lydia wants to go home
win first prize in the contest
freud wrote of the ego
I do not care if you do that
always cover all the bases
nobody cares anymore
can we play cards tonight
get rid of that immediately
I watched blazing saddles
the sum of the parts
they love to yap about nothing
peek out the window
be home before midnight
he played a pimp in that movie
I skimmed through your proposal
he was wearing a sweatshirt
no more war no more bloodshed
toss the ball around
I will meet you at noon
I want to hold your hand
the children are playing
superman never wore a mask
I listen to the tape everyday
he is shouting loudly
correct your diction immediately
seasoned golfers love the game
he cooled off after she left
my dog sheds his hair
join us on the patio
these cookies are so amazing
I can still feel your presence
the dog will bite you
a most ridiculous thing
where did you get that tie
what a lovely red jacket
do you like to shop on Sunday
I spilled coffee on the carpet
the largest of the five oceans
shall we play a round of cards
olympic athletes use drugs
my mother makes good cookies
do a good deed to someone
quick there is someone knocking
flashing red light means stop
sprawling subdivisions are bad
where did I leave my glasses
on the way to the cottage
a lot of chlorine in the water
do not drink the water
my car always breaks in the winter
santa claus got stuck
public transit is much faster
zero in on the facts
make up a few more phrases
my fingers are very cold
rain rain go away
bad for the environment
universities are too expensive
the price of gas is high
the winner of the race
we drive on parkways
we park in driveways
go out for some pizza and beer
effort is what it will take
where can my little dog be
if you were not so stupid
not quite so smart as you think
do you like to go camping
this person is a disaster
the imagination of the nation
universally understood to be wrong
listen to five hours of opera
an occasional taste of chocolate
victims deserve more redress
the protesters blocked all traffic
the acceptance speech was boring
work hard to reach the summit
a little encouragement is needed
stiff penalty for staying out late
the pen is mightier than the sword
exceed the maximum speed limit
in sharp contrast to your words
this leather jacket is too warm
consequences of a wrong turn
this mission statement is baloney
you will loose your voice
every apple from every tree
are you sure you want this
the fourth edition was better
this system of taxation
beautiful paintings in the gallery
a yard is almost as long as a meter
we missed your birthday
coalition governments never work
destruction of the rain forest
I like to play tennis
acutely aware of her good looks
you want to eat your cake
machinery is too complicated
a glance in the right direction
I just cannot figure this out
please follow the guidelines
an airport is a very busy place
mystery of the lost lagoon
is there any indication of this
the chamber makes important decisions
this phenomenon will never occur
obligations must be met first
valid until the end of the year
file all complaints in writing
tickets are very expensive
a picture is worth many words
this camera takes nice photographs
it looks like a shack
the dog buried the bone
the daring young man
this equation is too complicated
express delivery is very fast
I will put on my glasses
a touchdown in the last minute
the treasury department is broke
a good response to the question
well connected with people
the bathroom is good for reading
the generation gap gets wider
chemical spill took forever
prepare for the exam in advance
interesting observation was made
bank transaction was not registered
your etiquette needs some work
we better investigate this
stability of the nation
house with new electrical panel
our silver anniversary is coming
the presidential suite is very busy
the punishment should fit the crime
sharp cheese keeps the mind sharp
the registration period is over
you have my sympathy
the objective of the exercise
historic meeting without a result
very reluctant to enter
good at addition and subtraction
six daughters and seven sons
a thoroughly disgusting thing to say
sign the withdrawal slip
relations are very strained
the minimum amount of time
a very traditional way to dress
the aspirations of a nation
medieval times were very hard
a security force of eight thousand
there are winners and losers
the voters turfed him out
pay off a mortgage for a house
the collapse of the Roman empire
did you see that spectacular explosion
keep receipts for all your expenses
the assault took six months
get your priorities in order
traveling requires a lot of fuel
longer than a football field
a good joke deserves a good laugh
the union will go on strike
never mix religion and politics
interactions between men and women
where did you get such a silly idea
it should be sunny tomorrow
a psychiatrist will help you
you should visit to a doctor
you must make an appointment
the fax machine is broken
players must know all the rules
a dog is the best friend of a man
would you like to come to my house
February has an extra day
do not feel too bad about it
this library has many books
construction makes traveling difficult
he called seven times
that is a very odd question
a feeling of complete exasperation
we must redouble our efforts
no kissing in the library
that agreement is rife with problems
vote according to your conscience
my favourite sport is racketball
sad to hear that news
the gun discharged by accident
one of the poorest nations
the algorithm is too complicated
your presentation was inspiring
that land is owned by the government
burglars never leave their business card
the fire blazed all weekend
if diplomacy does not work
please keep this confidential
the rationale behind the decision
the cat has a pleasant temperament
our housekeeper does a thorough job
her majesty visited our country
handicapped persons need consideration
these barracks are big enough
sing the gospel and the blues
he underwent triple bypass surgery
the hopes of a new organization
peering through a small hole
rapidly running short on words
it is difficult to concentrate
give me one spoonful of coffee
two or three cups of coffee
just like it says on the can
companies announce a merger
electric cars need big fuel cells
the plug does not fit the socket
drugs should be avoided
the most beautiful sunset
we dine out on the weekends
get aboard the ship is leaving
the water was monitored daily
he watched in astonishment
a big scratch on the tabletop
salesmen must make their monthly quota
saving that child was an heroic effort
granite is the hardest of all rocks
bring the offenders to justice
every Saturday he folds the laundry
careless driving results in a fine
microscopes make small things look big
a coupon for a free sample
fine but only in moderation
a subject one can really enjoy
important for political parties
that sticker needs to be validated
the fire raged for an entire month
one never takes too many precautions
we have enough witnesses
labour unions know how to organize
people blow their horn a lot
a correction had to be published
I like baroque and classical music
the proprietor was unavailable
be discreet about your meeting
meet tomorrow in the lavatory
suburbs are sprawling up everywhere
shivering is one way to keep warm
dolphins leap high out of the water
try to enjoy your maternity leave
the ventilation system is broken
dinosaurs have been extinct for ages
an inefficient way to heat a house
the bus was very crowded
an injustice is committed every day
the coronation was very exciting
look in the syllabus for the course
rectangular objects have four sides
prescription drugs require a note
the insulation is not working
nothing finer than discovering a treasure
our life expectancy has increased
the cream rises to the top
the high waves will swamp us
the treasurer must balance her books
completely sold out of that
the location of the crime
the chancellor was very boring
the accident scene is a shrine for fans
a tumor is OK provided it is benign
please take a bath this month
rent is paid at the beginning of the month
for murder you get a long prison sentence
a much higher risk of getting cancer
quit while you are ahead
knee bone is connected to the thigh bone
safe to walk the streets in the evening
luckily my wallet was found
one hour is allotted for questions
so you think you deserve a raise
they watched the entire movie
good jobs for those with education
jumping right out of the water
the trains are always late
sit at the front of the bus
do you prefer a window seat
the food at this restaurant
Canada has ten provinces
the elevator door appears to be stuck
raindrops keep falling on my head
spill coffee on the carpet
an excellent way to communicate
with each step forward
faster than a speeding bullet
wishful thinking is fine
nothing wrong with his style
arguing with the boss is futile
taking the train is usually faster
what goes up must come down
be persistent to win a strike
presidents drive expensive cars
the stock exchange dipped
why do you ask silly questions
that is a very nasty cut
what to do when the oil runs dry
learn to walk before you run
insurance is important for bad drivers
traveling to conferences is fun
do you get nervous when you speak
pumping helps if the roads are slippery
parking tickets can be challenged
apartments are too expensive
find a nearby parking spot
gun powder must be handled with care
just what the doctor ordered
a rattle snake is very poisonous
weeping willows are found near water
I cannot believe I ate the whole thing
the biggest hamburger I have ever seen
gamblers eventually loose their shirts
exercise is good for the mind
irregular verbs are the hardest to learn
they might find your comment offensive
tell a lie and your nose will grow
an enlarged nose suggests you are a liar
lie detector tests never work
do not lie in court or else
most judges are very honest
only an idiot would lie in court
important news always seems to be late
please try to be home before midnight
if you come home late the doors are locked
dormitory doors are locked at midnight
staying up all night is a bad idea
you are a capitalist pig
motivational seminars make me sick
questioning the wisdom of the courts
rejection letters are discouraging
the first time he tried to swim
that referendum asked a silly question
a steep learning curve in riding a unicycle
a good stimulus deserves a good response
everybody looses in custody battles
put garbage in an abandoned mine
employee recruitment takes a lot of effort
experience is hard to come by
everyone wants to win the lottery
the picket line gives me the chills

"""


def grammar_info_gen(letter_labels, EVENT_LEN_S=3, separation_tok_in_word=False):
    if EVENT_LEN_S not in (3, 1.5):
        EVENT_LEN_S = 3

    if EVENT_LEN_S == 1.5:
        word_separation_tok = "  "
        sent_separation_tok = "   "
        stimulus_text_content = (
            (
                "\n".join(
                    [
                        "".join(p).replace(sent_separation_tok, "\n")[1:]
                        for p in letter_labels
                    ]
                )
            )
            .replace("\n\n", "\n")
            .replace(word_separation_tok, " ")
        )

        stimulus_letters = np.unique([l for l in stimulus_text_content])
        letter_conversion_dict = {l: l for l in stimulus_letters}

        if separation_tok_in_word:
            letter_conversion_dict["\n"] = "_space_"
            letter_conversion_dict[" "] = "_space_"

            letter_space_conversion_dict = {l: l for l in stimulus_letters}
            letter_space_conversion_dict["\n"] = " "
            letter_space_conversion_dict[" "] = " "

        else:
            letter_conversion_dict["\n"] = "_space_ _space_ _space_"
            letter_conversion_dict[" "] = "_space_ _space_"

            letter_space_conversion_dict = {l: l for l in stimulus_letters}
            letter_space_conversion_dict["\n"] = "   "
            letter_space_conversion_dict[" "] = "  "

    else:
        word_separation_tok = " "
        sent_separation_tok = "  "
        stimulus_text_content = (
            "\n".join(
                [
                    "".join(p).replace(sent_separation_tok, "\n")[1:]
                    for p in letter_labels
                ]
            )
        ).replace("\n\n", "\n")
        stimulus_letters = np.unique([l for l in stimulus_text_content])
        letter_conversion_dict = {l: l for l in stimulus_letters}

        if separation_tok_in_word:
            letter_conversion_dict["\n"] = "_space_"
            letter_conversion_dict[" "] = "_space_"

            letter_space_conversion_dict = {l: l for l in stimulus_letters}
            letter_space_conversion_dict["\n"] = " "
            letter_space_conversion_dict[" "] = " "

        else:
            letter_conversion_dict["\n"] = "_space_ _space_"
            letter_conversion_dict[" "] = "_space_"

            letter_space_conversion_dict = {l: l for l in stimulus_letters}
            letter_space_conversion_dict["\n"] = "  "
            letter_space_conversion_dict[" "] = " "

    stimulus_text_letter = "".join(
        [letter_space_conversion_dict[l] for l in stimulus_text_content]
    )

    space_tok = "_space_"
    unique_stimulus_words = np.unique(stimulus_text_content.split()).tolist()
    mackenzie_soukoreff_content = mackenzie_soukoreff_corpus.lower()
    unique_mackenzie_soukoreff_words = np.unique(
        mackenzie_soukoreff_content.split()
    ).tolist()
    # unique_stimulus_words += [" "]
    # unique_mackenzie_soukoreff_words += [" "]

    mackenzie_soukoreff_text_letter = "".join(
        [letter_space_conversion_dict[l] for l in mackenzie_soukoreff_content]
    )

    SENT_START_TOK = "SENT-START"
    SENT_END_TOK = "SENT-END"
    additional_key_val = {
        "SENT-END": [" "],
        "SENT-START": [" "],
        "_space_": [" "],
        "!NULL": [],
    }
    if separation_tok_in_word:
        additional_key_val["SENT-START"] = []

    unique_stimulus_word_dictionary = {
        w: [l for l in w if l != " "] for w in unique_stimulus_words
    }
    unique_mackenzie_soukoreff_word_dictionary = {
        w: [l for l in w if l != " "] for w in unique_mackenzie_soukoreff_words
    }
    if separation_tok_in_word:
        word_sep_list = [l for l in word_separation_tok]
        unique_stimulus_word_dictionary = {
            key: val + word_sep_list
            for key, val in unique_stimulus_word_dictionary.items()
        }
        unique_mackenzie_soukoreff_word_dictionary = {
            key: val + word_sep_list
            for key, val in unique_mackenzie_soukoreff_word_dictionary.items()
        }

    unique_stimulus_word_dictionary.update(additional_key_val)
    unique_mackenzie_soukoreff_word_dictionary.update(additional_key_val)

    # HTK grammar expression rules
    # $var = expression ; denotes defining sub expression
    # |                   denotes alternatives
    # [ ]                 encloses options
    # { }                 denotes zero or more repetitions
    # < >                 denotes one or more repetitions
    # << >>               denotes context-sensitive loop
    if separation_tok_in_word:
        sentences_formed_by_stimulus_words_seperated_by_space_dict_grammar = f'$word = {" | ".join([k for k in unique_stimulus_words if k not in [SENT_START_TOK, SENT_END_TOK, " "]])} ;\n({{_space_}}<{SENT_START_TOK} {{ $word }} {SENT_END_TOK}> {{_space_}} )'
        sentences_formed_by_mackenzie_soukoreff_words_seperated_by_space_dict_grammar = f'$word = {" | ".join([k for k in unique_mackenzie_soukoreff_words if k not in [SENT_START_TOK, SENT_END_TOK, " "]])} ;\n({{_space_}}<{SENT_START_TOK} {{ $word }} $word {SENT_END_TOK}> {{_space_}} )'
    else:
        sentences_formed_by_stimulus_words_seperated_by_space_dict_grammar = f'$word = {" | ".join([k for k in unique_stimulus_words if k not in [SENT_START_TOK, SENT_END_TOK, " "]])} ;\n(<{SENT_START_TOK} {{ $word _space_ }} $word {SENT_END_TOK}> {{_space_}} )'
        sentences_formed_by_mackenzie_soukoreff_words_seperated_by_space_dict_grammar = f'$word = {" | ".join([k for k in unique_mackenzie_soukoreff_words if k not in [SENT_START_TOK, SENT_END_TOK, " "]])} ;\n(<{SENT_START_TOK} {{ $word _space_ }} $word {SENT_END_TOK}> {{_space_}} )'

    sentences_formed_by_stimulus_words_seperated_by_space_lattice_string = (
        get_word_lattice_from_grammar(
            sentences_formed_by_stimulus_words_seperated_by_space_dict_grammar
        )
    )
    sentences_formed_by_mackenzie_soukoreff_words_seperated_by_space_lattice_string = get_word_lattice_from_grammar(
        sentences_formed_by_mackenzie_soukoreff_words_seperated_by_space_dict_grammar
    )
    stimulus_words_node_symbols, stimulus_words_link_start_end = parseLatticeString(
        sentences_formed_by_stimulus_words_seperated_by_space_lattice_string
    )
    (
        mackenzie_soukoreff_words_node_symbols,
        mackenzie_soukoreff_words_link_start_end,
    ) = parseLatticeString(
        sentences_formed_by_mackenzie_soukoreff_words_seperated_by_space_lattice_string
    )

    any_word_to_any_word_in_stimulus_dict_grammar = f'$word = {" | ".join([k for k in unique_stimulus_words if k not in [SENT_START_TOK, SENT_END_TOK, " "]])} | _space_ ;\n(<$word>)'
    any_word_to_any_word_in_mackenzie_soukoreff_dict_grammar = f'$word = {" | ".join([k for k in unique_mackenzie_soukoreff_words if k not in [SENT_START_TOK, SENT_END_TOK, " "]])} | _space_ ;\n(<$word>)'

    any_word_to_any_word_in_stimulus_lattice_string = get_word_lattice_from_grammar(
        any_word_to_any_word_in_stimulus_dict_grammar
    )
    any_word_to_any_word_in_mackenzie_soukoreff_lattice_string = (
        get_word_lattice_from_grammar(
            any_word_to_any_word_in_mackenzie_soukoreff_dict_grammar
        )
    )
    (
        aw2aw_stimulus_words_node_symbols,
        aw2aw_stimulus_words_link_start_end,
    ) = parseLatticeString(any_word_to_any_word_in_stimulus_lattice_string)
    (
        aw2aw_mackenzie_soukoreff_words_node_symbols,
        aw2aw_mackenzie_soukoreff_words_link_start_end,
    ) = parseLatticeString(any_word_to_any_word_in_mackenzie_soukoreff_lattice_string)

    unique_stimulus_word_dictionary_string = "\n".join(
        [
            f'{word} {" ".join(spelling) if spelling != [" "] else "_space_"}'
            for word, spelling in unique_stimulus_word_dictionary.items()
            if (word != " ") and (word != "!NULL")
        ]
    )
    unique_mackenzie_soukoreff_word_dictionary_string = "\n".join(
        [
            f'{word} {" ".join(spelling) if spelling != [" "] else "_space_"}'
            for word, spelling in unique_mackenzie_soukoreff_word_dictionary.items()
            if (word != " ") and (word != "!NULL")
        ]
    )

    mackenzie_soukoreff_content_letters = (
        "_space_ "
        + " ".join(mackenzie_soukoreff_content)
        .replace("  ", " _space_")
        .replace("\n", "_space_ _space_")
        + " _space_ _space_"
    )

    mackenzie_soukoreff_letter_one_gram_count = get_one_gram_feat_vector(
        mackenzie_soukoreff_text_letter
    )
    mackenzie_soukoreff_letter_bigram_count = get_two_gram_feat_vector(
        mackenzie_soukoreff_text_letter
    )
    stimulus_letter_one_gram_count = get_one_gram_feat_vector(stimulus_text_letter)
    stimulus_letter_bigram_count = get_two_gram_feat_vector(stimulus_text_letter)

    return {
        "word_separation_tok": word_separation_tok,
        "sent_separation_tok": sent_separation_tok,
        "stimulus_text_content": stimulus_text_content,
        "stimulus_letters": stimulus_letters,
        "letter_conversion_dict": letter_conversion_dict,
        "letter_conversion_dict": letter_conversion_dict,
        "letter_conversion_dict": letter_conversion_dict,
        "letter_space_conversion_dict": letter_space_conversion_dict,
        "letter_space_conversion_dict": letter_space_conversion_dict,
        "letter_space_conversion_dict": letter_space_conversion_dict,
        "stimulus_text_letter": stimulus_text_letter,
        "space_tok": space_tok,
        "unique_stimulus_words": unique_stimulus_words,
        "mackenzie_soukoreff_content": mackenzie_soukoreff_content,
        "unique_mackenzie_soukoreff_words": unique_mackenzie_soukoreff_words,
        "unique_stimulus_words": unique_stimulus_words,
        "unique_mackenzie_soukoreff_words": unique_mackenzie_soukoreff_words,
        "mackenzie_soukoreff_text_letter": mackenzie_soukoreff_text_letter,
        "SENT_START_TOK": SENT_START_TOK,
        "SENT_END_TOK": SENT_END_TOK,
        "additional_key_val": additional_key_val,
        "unique_stimulus_word_dictionary": unique_stimulus_word_dictionary,
        "unique_mackenzie_soukoreff_word_dictionary": unique_mackenzie_soukoreff_word_dictionary,
        "unique_stimulus_word_dictionary": unique_stimulus_word_dictionary,
        "unique_mackenzie_soukoreff_word_dictionary": unique_mackenzie_soukoreff_word_dictionary,
        "sentences_formed_by_stimulus_words_seperated_by_space_dict_grammar": sentences_formed_by_stimulus_words_seperated_by_space_dict_grammar,
        "sentences_formed_by_mackenzie_soukoreff_words_seperated_by_space_dict_grammar": sentences_formed_by_mackenzie_soukoreff_words_seperated_by_space_dict_grammar,
        "sentences_formed_by_stimulus_words_seperated_by_space_lattice_string": sentences_formed_by_stimulus_words_seperated_by_space_lattice_string,
        "sentences_formed_by_mackenzie_soukoreff_words_seperated_by_space_lattice_string": sentences_formed_by_mackenzie_soukoreff_words_seperated_by_space_lattice_string,
        "stimulus_words_node_symbols": stimulus_words_node_symbols,
        "stimulus_words_link_start_end": stimulus_words_link_start_end,
        "mackenzie_soukoreff_words_node_symbols": mackenzie_soukoreff_words_node_symbols,
        "mackenzie_soukoreff_words_link_start_end": mackenzie_soukoreff_words_link_start_end,
        "any_word_to_any_word_in_stimulus_dict_grammar": any_word_to_any_word_in_stimulus_dict_grammar,
        "any_word_to_any_word_in_mackenzie_soukoreff_dict_grammar": any_word_to_any_word_in_mackenzie_soukoreff_dict_grammar,
        "any_word_to_any_word_in_stimulus_lattice_string": any_word_to_any_word_in_stimulus_lattice_string,
        "any_word_to_any_word_in_mackenzie_soukoreff_lattice_string": any_word_to_any_word_in_mackenzie_soukoreff_lattice_string,
        "aw2aw_stimulus_words_node_symbols": aw2aw_stimulus_words_node_symbols,
        "aw2aw_stimulus_words_link_start_end": aw2aw_stimulus_words_link_start_end,
        "aw2aw_mackenzie_soukoreff_words_node_symbols": aw2aw_mackenzie_soukoreff_words_node_symbols,
        "aw2aw_mackenzie_soukoreff_words_link_start_end": aw2aw_mackenzie_soukoreff_words_link_start_end,
        "unique_stimulus_word_dictionary_string": unique_stimulus_word_dictionary_string,
        "unique_mackenzie_soukoreff_word_dictionary_string": unique_mackenzie_soukoreff_word_dictionary_string,
        "mackenzie_soukoreff_content_letters": mackenzie_soukoreff_content_letters,
        "mackenzie_soukoreff_letter_one_gram_count": mackenzie_soukoreff_letter_one_gram_count,
        "mackenzie_soukoreff_letter_bigram_count": mackenzie_soukoreff_letter_bigram_count,
        "stimulus_letter_one_gram_count": stimulus_letter_one_gram_count,
        "stimulus_letter_bigram_count": stimulus_letter_bigram_count,
    }
