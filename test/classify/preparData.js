const request = require("request-promise");
const fs = require("fs");
const os = require("os");
(async () => {
    let features = ["h", "d", "a", "fixedodds", "result"];
    let DataStr = features.join() + "\n";
    for (let matchid = 93527; matchid < 104528; matchid++) {
        try {
            let r = JSON.parse(
                await request(
                    `http://i.sporttery.cn/api/fb_match_info/get_pool_rs/?mid=${matchid}`
                )
            );
            if (r.status.code !== 0 || !r.result.pool_rs) continue;
            let tmp = r.result.odds_list.had.odds.slice(-1)[0];
            let ar = [
                +tmp.h,
                +tmp.d,
                +tmp.a,
                +r.result.odds_list.hhad.goalline,
                "hda".indexOf(r.result.pool_rs.had.pool_rs)
            ];
            console.log(matchid, ar);
            DataStr += ar.join();
            DataStr += "\n";
            // console.log(ar);
        } catch (error) {
            continue;
        }
    }

    fs.writeFileSync("data/data2.csv", DataStr);
})();
