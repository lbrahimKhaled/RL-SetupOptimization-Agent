-- RL Teleport Test (CSP Lua app). Tests three things the training start needs:
--   1. physics.setCarPosition(0, pos, dir): car placed at a saved spot, aligned on the road
--   2. physics.setCarVelocity(0, v): car released already moving (30 m/s along its heading)
--   3. the file channel: an outside program writes request.txt, this app teleports and writes ack.txt
-- Every CSP call is wrapped in pcall; failures are shown in the window instead of failing silently.

local saved = nil            -- {pos = vec3, dir = vec3}
local status = 'ready: drive somewhere, press "Save spot"'
local lastRequest = nil
local pollAccum = 0
local dir = __dirname or '.'
local requestFile = dir .. '/request.txt'
local ackFile = dir .. '/ack.txt'
local savedFile = dir .. '/saved.txt'

local function fmt(v) return string.format('%.2f %.2f %.2f', v.x, v.y, v.z) end

local function teleport(pos, look, speed)
  local ok, err = pcall(physics.setCarPosition, 0, pos, look)
  if not ok then return false, 'setCarPosition FAILED: ' .. tostring(err) end
  if speed and speed > 0 then
    local n = math.sqrt(look.x * look.x + look.y * look.y + look.z * look.z)
    if n <= 0 then return false, 'zero direction' end
    local v = vec3(look.x / n * speed, look.y / n * speed, look.z / n * speed)
    local ok2, err2 = pcall(physics.setCarVelocity, 0, v)
    if not ok2 then return false, 'setCarPosition OK, setCarVelocity FAILED: ' .. tostring(err2) end
    return true, string.format('teleported, released at %.0f m/s', speed)
  end
  return true, 'teleported (stopped)'
end

-- request.txt: one line "<id> <x> <y> <z> <dx> <dy> <dz> <speed_mps>"; a new id triggers one teleport.
local function pollRequest()
  local ok, text = pcall(io.load, requestFile, nil)
  if not ok then status = 'io.load FAILED: ' .. tostring(text); return end
  if not text or text == '' or text == lastRequest then return end
  lastRequest = text
  local f = {}
  for w in text:gmatch('%S+') do f[#f + 1] = w end
  local id = f[1] or '?'
  local nums = {}
  for i = 2, 8 do nums[#nums + 1] = tonumber(f[i]) end
  local result
  if #nums ~= 7 then
    result = 'bad request: expected id + 7 numbers'
  else
    local good, msg = teleport(vec3(nums[1], nums[2], nums[3]), vec3(nums[4], nums[5], nums[6]), nums[7])
    result = (good and 'ok ' or 'error ') .. msg
  end
  status = 'request ' .. id .. ': ' .. result
  local okSave, errSave = pcall(io.save, ackFile, id .. ' ' .. result)
  if not okSave then status = status .. ' | io.save FAILED: ' .. tostring(errSave) end
end

function windowMain(dt)
  local okCar, car = pcall(ac.getCar, 0)
  if not okCar or not car then
    ui.text('ac.getCar(0) FAILED: ' .. tostring(car))
    return
  end
  ui.text(string.format('Speed: %.1f km/h', car.speedKmh or -1))
  ui.text('Position: ' .. fmt(car.position))
  ui.text('Heading:  ' .. fmt(car.look))
  ui.separator()

  if ui.button('Save spot') then
    saved = {pos = vec3(car.position.x, car.position.y, car.position.z),
             dir = vec3(car.look.x, car.look.y, car.look.z)}
    status = 'saved: ' .. fmt(saved.pos)
    pcall(io.save, savedFile, fmt(saved.pos) .. ' ' .. fmt(saved.dir) .. ' 0')
  end
  if saved then
    if ui.button('Teleport to saved spot (stopped)') then
      local _, msg = teleport(saved.pos, saved.dir, 0)
      status = msg
    end
    if ui.button('Teleport to saved spot at 30 m/s') then
      local _, msg = teleport(saved.pos, saved.dir, 30)
      status = msg
    end
    ui.text('Saved: ' .. fmt(saved.pos))
  end
  ui.separator()
  ui.text('Status:')
  for i = 1, #status, 50 do ui.text(status:sub(i, i + 49)) end
  ui.text('Request file: request.txt in this app folder')

  local now = os.clock()
  if now - pollAccum >= .25 then
    pollAccum = now
    pollRequest()
  end
end
